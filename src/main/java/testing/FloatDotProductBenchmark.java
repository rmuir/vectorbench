package testing;

import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.file.StandardOpenOption.*;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector", "--enable-preview"})
public class FloatDotProductBenchmark {

  private float[] a;
  private float[] b;

  private MemorySegment segment;

  @Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100", "128", "207", "256", "300", "512", "702", "1024"})
  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100" })
  //@Param({"702", "1024"})
  //@Param({"16", "32", "64"})
  int size;

  @Setup(Level.Trial)
  public void init() throws IOException {
    float[] tmpA = new float[size];
    float[] tmpB = new float[size];
    for (int i = 0; i < size; ++i) {
      tmpA[i] = ThreadLocalRandom.current().nextFloat();
      tmpB[i] = ThreadLocalRandom.current().nextFloat();
    }
    Path p = Path.of("vector.data");
    try (FileChannel fc = FileChannel.open(p, CREATE, READ, WRITE)) {
      ByteBuffer buf = ByteBuffer.allocate(size * 2 * Float.BYTES);
      buf.order(LITTLE_ENDIAN);
      buf.asFloatBuffer().put(0, tmpA);
      buf.asFloatBuffer().put(size, tmpB);
      int n = fc.write(buf);
      assert n == size * 2 * Float.BYTES;

      Arena arena = Arena.openShared();
      segment = fc.map(FileChannel.MapMode.READ_ONLY, 0, size * 2L * Float.BYTES, arena.scope());
    }

    // Thread local buffers
    a = new float[size];
    b = new float[size];

    float f1 = dotProductCopyFromArray();
    float f2 = dotProductFromMemorySegment();
    if (Math.abs(f1 - f2) > 0.0001) {
      throw new AssertionError(f1 + " != " + f2);
    }
  }

  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  static final ValueLayout.OfFloat LAYOUT_LE_FLOAT =
          ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Benchmark
  public float dotProductCopyFromArray() {
    // loads vector data from the backing memory segment into the float arrays,
    // in the same way as that of Lucene's MemorySegmentIndexInput.
    MemorySegment.copy(segment, LAYOUT_LE_FLOAT, 0, a, 0, size);
    MemorySegment.copy(segment, LAYOUT_LE_FLOAT, (long) size * Float.BYTES, b, 0, size);

    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    int i = 0;
    float res = 0;
    // if the array size is large (> 2x platform vector size), its worth the overhead to vectorize
    if (a.length > 2 * SPECIES.length()) {
      // vector loop is unrolled 4x (4 accumulators in parallel)
      FloatVector acc1 = FloatVector.zero(SPECIES);
      FloatVector acc2 = FloatVector.zero(SPECIES);
      FloatVector acc3 = FloatVector.zero(SPECIES);
      FloatVector acc4 = FloatVector.zero(SPECIES);
      int upperBound = SPECIES.loopBound(a.length - 3*SPECIES.length());
      for (; i < upperBound; i += 4 * SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        acc1 = acc1.add(va.mul(vb));
        FloatVector vc = FloatVector.fromArray(SPECIES, a, i + SPECIES.length());
        FloatVector vd = FloatVector.fromArray(SPECIES, b, i + SPECIES.length());
        acc2 = acc2.add(vc.mul(vd));
        FloatVector ve = FloatVector.fromArray(SPECIES, a, i + 2*SPECIES.length());
        FloatVector vf = FloatVector.fromArray(SPECIES, b, i + 2*SPECIES.length());
        acc3 = acc3.add(ve.mul(vf));
        FloatVector vg = FloatVector.fromArray(SPECIES, a, i + 3*SPECIES.length());
        FloatVector vh = FloatVector.fromArray(SPECIES, b, i + 3*SPECIES.length());
        acc4 = acc4.add(vg.mul(vh));
      }
      // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
      upperBound = SPECIES.loopBound(a.length);
      for (; i < upperBound; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        acc1 = acc1.add(va.mul(vb));
      }
      // reduce
      FloatVector res1 = acc1.add(acc2);
      FloatVector res2 = acc3.add(acc4);
      res += res1.add(res2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      res += b[i] * a[i];
    }
    return res;
  }

  @Benchmark
  public float dotProductFromMemorySegment() {
    return dotProductFromMemorySegment(0, size * Float.BYTES);
  }

  private float dotProductFromMemorySegment(long segmentOffset1, long segmentOffset2) {
    // Here we are using a single memory segment that holds the data for both vectors.
    // This is similar to what Lucene's MemorySegmentIndexInput would expose (rather
    // than a segment per vector. since segments are immutable and we would not want
    // the garbage).
    if (segment.byteSize() < size * Float.BYTES * 2L) {
      throw new IllegalArgumentException("segment too small");
    }
    int i = 0;
    float res = 0;
    // if the array size is large (> 2x platform vector size), its worth the overhead to vectorize
    final int length = size; // good enough for benching, but this would be passed
    if (length > 2 * SPECIES.length()) {
      // vector loop is unrolled 4x (4 accumulators in parallel)
      FloatVector acc1 = FloatVector.zero(SPECIES);
      FloatVector acc2 = FloatVector.zero(SPECIES);
      FloatVector acc3 = FloatVector.zero(SPECIES);
      FloatVector acc4 = FloatVector.zero(SPECIES);
      int upperBound = SPECIES.loopBound(length - 3*SPECIES.length());
      for (; i < upperBound; i += 4 * SPECIES.length()) {
        // int offset = SPECIES.length() * Float.BYTES;
        FloatVector va = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset1 + i * Float.BYTES, LITTLE_ENDIAN);
        FloatVector vb = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset2 + i * Float.BYTES, LITTLE_ENDIAN);
        acc1 = acc1.add(va.mul(vb));
        FloatVector vc = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset1 + (i + SPECIES.length()) * Float.BYTES, LITTLE_ENDIAN);
        FloatVector vd = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset2 + (i + SPECIES.length()) * Float.BYTES, LITTLE_ENDIAN);
        acc2 = acc2.add(vc.mul(vd));
        FloatVector ve = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset1 + (i + 2*SPECIES.length()) * Float.BYTES, LITTLE_ENDIAN);
        FloatVector vf = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset2 +(i + 2*SPECIES.length()) * Float.BYTES, LITTLE_ENDIAN);
        acc3 = acc3.add(ve.mul(vf));
        FloatVector vg = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset1 + (i + 3*SPECIES.length()) * Float.BYTES, LITTLE_ENDIAN);
        FloatVector vh = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset2 + (i + 3*SPECIES.length()) * Float.BYTES, LITTLE_ENDIAN);
        acc4 = acc4.add(vg.mul(vh));
      }
      // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
      upperBound = SPECIES.loopBound(length);
      for (; i < upperBound; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset1 + i * Float.BYTES, LITTLE_ENDIAN);
        FloatVector vb = FloatVector.fromMemorySegment(SPECIES, segment, segmentOffset2 + i * Float.BYTES, LITTLE_ENDIAN);
        acc1 = acc1.add(va.mul(vb));
      }
      // reduce
      FloatVector res1 = acc1.add(acc2);
      FloatVector res2 = acc3.add(acc4);
      res += res1.add(res2).reduceLanes(VectorOperators.ADD);
    }

    // tail
    for (; i < length; i++) {
      res += segment.get(LAYOUT_LE_FLOAT, segmentOffset2 + i * Float.BYTES) * segment.get(LAYOUT_LE_FLOAT, segmentOffset1 + i * Float.BYTES);
    }
    return res;
  }

  @Benchmark
  public float dotProductOld() {
    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    float res = 0f;
    /*
     * If length of vector is larger than 8, we use unrolled dot product to accelerate the
     * calculation.
     */
    int i;
    for (i = 0; i < a.length % 8; i++) {
      res += b[i] * a[i];
    }
    if (a.length < 8) {
      return res;
    }
    for (; i + 31 < a.length; i += 32) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
      res +=
          b[i + 8] * a[i + 8]
              + b[i + 9] * a[i + 9]
              + b[i + 10] * a[i + 10]
              + b[i + 11] * a[i + 11]
              + b[i + 12] * a[i + 12]
              + b[i + 13] * a[i + 13]
              + b[i + 14] * a[i + 14]
              + b[i + 15] * a[i + 15];
      res +=
          b[i + 16] * a[i + 16]
              + b[i + 17] * a[i + 17]
              + b[i + 18] * a[i + 18]
              + b[i + 19] * a[i + 19]
              + b[i + 20] * a[i + 20]
              + b[i + 21] * a[i + 21]
              + b[i + 22] * a[i + 22]
              + b[i + 23] * a[i + 23];
      res +=
          b[i + 24] * a[i + 24]
              + b[i + 25] * a[i + 25]
              + b[i + 26] * a[i + 26]
              + b[i + 27] * a[i + 27]
              + b[i + 28] * a[i + 28]
              + b[i + 29] * a[i + 29]
              + b[i + 30] * a[i + 30]
              + b[i + 31] * a[i + 31];
    }
    for (; i + 7 < a.length; i += 8) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
    }
    return res;
  }
}
