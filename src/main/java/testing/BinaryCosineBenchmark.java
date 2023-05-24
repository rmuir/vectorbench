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
public class BinaryCosineBenchmark {

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
    if (cosineDistanceNew() != cosineDistanceOld()) {
      throw new RuntimeException("New is wrong");
    }
  }

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

  private static final boolean IS_AMD64_WITHOUT_AVX2 =
    System.getProperty("os.arch").equals("amd64") && IntVector.SPECIES_PREFERRED.vectorBitSize() < 256;

  @Benchmark
  public float cosineDistanceNew() {
    int i = 0;
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;
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
        IntVector accSum = IntVector.zero(IntVector.SPECIES_PREFERRED);
        IntVector accNorm1 = IntVector.zero(IntVector.SPECIES_PREFERRED);
        IntVector accNorm2 = IntVector.zero(IntVector.SPECIES_PREFERRED);
        for (; i < upperBound; i += PREFERRED_BYTE_SPECIES.length()) {
          ByteVector va8 = ByteVector.fromArray(PREFERRED_BYTE_SPECIES, a, i);
          ByteVector vb8 = ByteVector.fromArray(PREFERRED_BYTE_SPECIES, b, i);
          Vector<Short> va16 = va8.convertShape(VectorOperators.B2S, PREFERRED_SHORT_SPECIES, 0);
          Vector<Short> vb16 = vb8.convertShape(VectorOperators.B2S, PREFERRED_SHORT_SPECIES, 0);
          Vector<Short> prod16 = va16.mul(vb16);
          Vector<Short> norm1_16 = va16.mul(va16);
          Vector<Short> norm2_16 = vb16.mul(vb16);
          Vector<Integer> prod32 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0);
          Vector<Integer> norm1_32 = norm1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0);
          Vector<Integer> norm2_32 = norm2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0);
          accSum = accSum.add(prod32);
          accNorm1 = accNorm1.add(norm1_32);
          accNorm2 = accNorm2.add(norm2_32);
        }
        // reduce
        sum += accSum.reduceLanes(VectorOperators.ADD);
        norm1 += accNorm1.reduceLanes(VectorOperators.ADD);
        norm2 += accNorm2.reduceLanes(VectorOperators.ADD);
      } else {
        // 128-bit implementation, which must "split up" vectors due to widening conversions
        int upperBound = ByteVector.SPECIES_64.loopBound(a.length);
        IntVector accSum1 = IntVector.zero(IntVector.SPECIES_128);
        IntVector accSum2 = IntVector.zero(IntVector.SPECIES_128);
        IntVector accNorm1_1 = IntVector.zero(IntVector.SPECIES_128);
        IntVector accNorm1_2 = IntVector.zero(IntVector.SPECIES_128);
        IntVector accNorm2_1 = IntVector.zero(IntVector.SPECIES_128);
        IntVector accNorm2_2 = IntVector.zero(IntVector.SPECIES_128);
        for (; i < upperBound; i += ByteVector.SPECIES_64.length()) {
          ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
          ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
          // expand each byte vector into short vector and perform multiplications
          Vector<Short> va16 = va8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> vb16 = vb8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> prod16 = va16.mul(vb16);
          Vector<Short> norm1_16 = va16.mul(va16);
          Vector<Short> norm2_16 = vb16.mul(vb16);
          // split each short vector into two int vectors and add
          Vector<Integer> prod32_1 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
          Vector<Integer> prod32_2 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
          Vector<Integer> norm1_32_1 = norm1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
          Vector<Integer> norm1_32_2 = norm1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
          Vector<Integer> norm2_32_1 = norm2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
          Vector<Integer> norm2_32_2 = norm2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
          accSum1 = accSum1.add(prod32_1);
          accSum2 = accSum2.add(prod32_2);
          accNorm1_1 = accNorm1_1.add(norm1_32_1);
          accNorm1_2 = accNorm1_2.add(norm1_32_2);
          accNorm2_1 = accNorm2_1.add(norm2_32_1);
          accNorm2_2 = accNorm2_2.add(norm2_32_2);
        }
        // reduce
        sum += accSum1.add(accSum2).reduceLanes(VectorOperators.ADD);
        norm1 += accNorm1_1.add(accNorm1_2).reduceLanes(VectorOperators.ADD);
        norm2 += accNorm2_1.add(accNorm2_2).reduceLanes(VectorOperators.ADD);
      }
    }

    for (; i < a.length; i++) {
      byte elem1 = a[i];
      byte elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  /** Returns the cosine similarity between the two vectors. */
  @Benchmark
  public float cosineDistanceOld() {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;

    for (int i = 0; i < a.length; i++) {
      byte elem1 = a[i];
      byte elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }
}
