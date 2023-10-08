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
public class BinaryDotProductBenchmark {

  private byte[] a;
  private byte[] b;

  //@Param({"1", "128", "207", "256", "300", "512", "702", "1024"})
  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100" })
  @Param({"1024"})
  //@Param({"16", "32", "64"})
  int size;

  private static final boolean IS_AMD64_WITHOUT_AVX2 =
      System.getProperty("os.arch").equals("amd64") && IntVector.SPECIES_PREFERRED.vectorBitSize() < 256;

  @Setup(Level.Trial)
  public void init() {
    a = new byte[size];
    b = new byte[size];
    ThreadLocalRandom.current().nextBytes(a);
    ThreadLocalRandom.current().nextBytes(b);
    if (dotProductNew() != dotProductOld()) {
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

  @Benchmark
  public int dotProductNewNew() {
    int i = 0;
    int res = 0;
    final int vectorSize = IntVector.SPECIES_PREFERRED.vectorBitSize();
    // only vectorize if we'll at least enter the loop a single time, and we have at least 128-bit vectors
    if (a.length >= 16 && vectorSize >= 128 && IS_AMD64_WITHOUT_AVX2 == false) {
      // compute vectorized dot product consistent with VPDPBUSD instruction, acts like:
      // int sum = 0;
      // for (...) {
      //   short product = (short) (x[i] * y[i]);
      //   sum += product;
      // }
      if (vectorSize >= 256) {
        // optimized 256/512 bit implementation, processes 8/16 bytes at a time
        int upperBound = PREFERRED_BYTE_SPECIES.loopBound(a.length);
        IntVector acc = IntVector.zero(IntVector.SPECIES_PREFERRED);
        for (; i < upperBound; i += PREFERRED_BYTE_SPECIES.length()) {
          ByteVector va8 = ByteVector.fromArray(PREFERRED_BYTE_SPECIES, a, i);
          ByteVector vb8 = ByteVector.fromArray(PREFERRED_BYTE_SPECIES, b, i);
          // widen to 32 bits, multiply and add
          Vector<Integer> va32 = va8.convertShape(VectorOperators.B2I, IntVector.SPECIES_PREFERRED, 0);
          Vector<Integer> vb32 = vb8.convertShape(VectorOperators.B2I, IntVector.SPECIES_PREFERRED, 0);
          acc = acc.add(va32.mul(vb32));
        }
        // reduce
        res += acc.reduceLanes(VectorOperators.ADD);
      } else {
        // 128-bit impl, which is tricky since we don't have SPECIES_32, it splits vectors up
        int upperBound = ByteVector.SPECIES_128.loopBound(a.length);
        IntVector acc = IntVector.zero(IntVector.SPECIES_128);
        for (; i < upperBound; i += ByteVector.SPECIES_128.length()) {
          ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
          ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);

          // first half
          Vector<Short> va16_1 = va8.convert(VectorOperators.B2S, 0);
          Vector<Short> vb16_1 = vb8.convert(VectorOperators.B2S, 0);
          Vector<Short> prod16_1 = va16_1.mul(vb16_1);

          // second half
          Vector<Short> va16_2 = va8.convert(VectorOperators.B2S, 1);
          Vector<Short> vb16_2 = vb8.convert(VectorOperators.B2S, 1);
          Vector<Short> prod16_2 = va16_2.mul(vb16_2);

          // sum into accumulators
          Vector<Short> prod16 = prod16_1.add(prod16_2);
          acc = acc.add(prod16.convert(VectorOperators.S2I, 0));
          acc = acc.add(prod16.convert(VectorOperators.S2I, 1));
        }
        // reduce
        res += acc.reduceLanes(VectorOperators.ADD);
      }
    }

    for (; i < a.length; i++) {
      res += b[i] * a[i];
    }
    return res;
  }

  @Benchmark
  public int dotProductNew() {
    int i = 0;
    int res = 0;
    final int vectorSize = IntVector.SPECIES_PREFERRED.vectorBitSize();
    // only vectorize if we'll at least enter the loop a single time, and we have at least 128-bit vectors
    if (a.length >= 16 && vectorSize >= 128 && IS_AMD64_WITHOUT_AVX2 == false) {
      // compute vectorized dot product consistent with VPDPBUSD instruction, acts like:
      // int sum = 0;
      // for (...) {
      //   short product = (short) (x[i] * y[i]);
      //   sum += product;
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
          Vector<Short> prod16 = va16.mul(vb16);
          Vector<Integer> prod32 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0);
          acc = acc.add(prod32);
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
          // expand each byte vector into short vector and multiply
          Vector<Short> va16 = va8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> vb16 = vb8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> prod16 = va16.mul(vb16);
          // split each short vector into two int vectors and add
          Vector<Integer> prod32_1 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
          Vector<Integer> prod32_2 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
          acc1 = acc1.add(prod32_1);
          acc2 = acc2.add(prod32_2);
        }
        // reduce
        res += acc1.add(acc2).reduceLanes(VectorOperators.ADD);
      }
    }

    for (; i < a.length; i++) {
      res += b[i] * a[i];
    }
    return res;
  }

  /**
   * Dot product computed over signed bytes.
   *
   * @param a bytes containing a vector
   * @param b bytes containing another vector, of the same dimension
   * @return the value of the dot product of the two vectors
   */
  @Benchmark
  public int dotProductOld() {
    assert a.length == b.length;
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += a[i] * b[i];
    }
    return total;
  }
}
