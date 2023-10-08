package testing;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.Vector;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
public class BinarySquareBenchmark {

  private byte[] a;
  private byte[] b;

  @Param({"1", "128", "207", "256", "300", "512", "702", "1024"})
  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100" })
  //@Param({"1024"})
  //@Param({"16", "32", "64"})
  int size;

  @Setup(Level.Trial)
  public void init() {
    a = new byte[size];
    b = new byte[size];
    ThreadLocalRandom.current().nextBytes(a);
    ThreadLocalRandom.current().nextBytes(b);
    if (squareDistanceNew() != squareDistanceOld()) {
      throw new RuntimeException("New is wrong");
    }
  }

  private static final boolean IS_AMD64_WITHOUT_AVX2 =
    System.getProperty("os.arch").equals("amd64") && IntVector.SPECIES_PREFERRED.vectorBitSize() < 256;

  static final VectorSpecies<Byte>  PREFERRED_BYTE_SPECIES;
  static final VectorSpecies<Short> PREFERRED_SHORT_SPECIES;
  static {
    if (IntVector.SPECIES_PREFERRED.vectorBitSize() >= 256) {
      PREFERRED_BYTE_SPECIES = ByteVector.SPECIES_MAX.withShape(VectorShape.forBitSize(IntVector.SPECIES_PREFERRED.vectorBitSize() >> 2));
      PREFERRED_SHORT_SPECIES = ShortVector.SPECIES_MAX.withShape(VectorShape.forBitSize(IntVector.SPECIES_PREFERRED.vectorBitSize() >> 1));
    } else {
      PREFERRED_BYTE_SPECIES = null;
      PREFERRED_SHORT_SPECIES = null;
    }
  }

  @Benchmark
  public int squareDistanceNew() {
    int i = 0;
    int res = 0;
    final int vectorSize = IntVector.SPECIES_PREFERRED.vectorBitSize();
    // only vectorize if we'll at least enter the loop a single time, and we have at least 128-bit vectors
    if (a.length >= 16 && vectorSize >= 128 && IS_AMD64_WITHOUT_AVX2 == false) {
      // acts like:
      // int sum = 0;
      // for (...) {
      //   short difference = (short) (x[i] - y[i]);
      //   sum += (int) difference * (int) difference;
      // }
      if (vectorSize >= 256) {
        // optimized 256/512 bit implementation, processes 8/16 bytes at a time
        int upperBound = PREFERRED_BYTE_SPECIES.loopBound(a.length);
        IntVector acc = IntVector.zero(IntVector.SPECIES_PREFERRED);
        for (; i < upperBound; i += PREFERRED_BYTE_SPECIES.length()) {
          ByteVector va8 = ByteVector.fromArray(PREFERRED_BYTE_SPECIES, a, i);
          ByteVector vb8 = ByteVector.fromArray(PREFERRED_BYTE_SPECIES, b, i);
          Vector<Short> va16 = va8.convertShape(VectorOperators.B2S, PREFERRED_SHORT_SPECIES, 0);
          Vector<Short> vb16 = vb8.convertShape(VectorOperators.B2S, PREFERRED_SHORT_SPECIES, 0);
          Vector<Short> diff16 = va16.sub(vb16);
          Vector<Integer> diff32 = diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0);
          acc = acc.add(diff32.mul(diff32));
        }
        // reduce
        res += acc.reduceLanes(VectorOperators.ADD);
      } else {
        // 128-bit implementation, which must "split up" vectors due to widening conversions
        int upperBound = ByteVector.SPECIES_64.loopBound(a.length);
        IntVector acc1 = IntVector.zero(IntVector.SPECIES_128);
        IntVector acc2 = IntVector.zero(IntVector.SPECIES_128);
        for (; i < upperBound; i += ByteVector.SPECIES_64.length()) {
          ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
          ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
          // expand each byte vector into short vector and subtract
          Vector<Short> va16 = va8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> vb16 = vb8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> diff16 = va16.sub(vb16);
          // split each short vector into two int vectors, square, and add
          Vector<Integer> diff32_1 = diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
          Vector<Integer> diff32_2 = diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
          acc1 = acc1.add(diff32_1.mul(diff32_1));
          acc2 = acc2.add(diff32_2.mul(diff32_2));
        }
        // reduce
        res += acc1.add(acc2).reduceLanes(VectorOperators.ADD);
      }
    }

    for (; i < a.length; i++) {
      int diff = a[i] - b[i];
      res += diff * diff;
    }
    return res;
  }

  /** Returns the sum of squared differences of the two vectors. */
  @Benchmark
  public int squareDistanceOld() {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int squareSum = 0;
    for (int i = 0; i < a.length; i++) {
      int diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }
}
