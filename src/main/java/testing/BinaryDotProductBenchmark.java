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
    if (dotProductNew() != dotProductOld()) {
      throw new RuntimeException("wrong");
    }
  }

  @Benchmark
  public int dotProductNew() {
    int i = 0;
    int res = 0;
    final int vectorSize = IntVector.SPECIES_PREFERRED.vectorBitSize();
    // only vectorize if we'll at least enter the loop a single time, and we have at least 128-bit vectors
    if (a.length >= ByteVector.SPECIES_64.length() && vectorSize >= 128) {
      // compute vectorized dot product consistent with VPDPBUSD instruction, acts like:
      // int sum = 0;
      // for (...) {
      //   short product = (short) (x[i] * y[i]);
      //   sum += product;
      // }
      if (vectorSize >= 256) {
        // optimized 256 bit implementation, processes 8 bytes at a time
        int upperBound = ByteVector.SPECIES_64.loopBound(a.length);
        IntVector acc = IntVector.zero(IntVector.SPECIES_256);
        for (; i < upperBound; i += ByteVector.SPECIES_64.length()) {
          ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
          ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
          Vector<Short> va16 = va8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> vb16 = vb8.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
          Vector<Short> prod16 = va16.mul(vb16);
          Vector<Integer> prod32 = prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
          acc = acc.add(prod32);
        }
        // reduce
        res += acc.reduceLanes(VectorOperators.ADD);
      } else {
        // 128-bit implementation
        // generic implementation, which must "split up" vectors due to widening conversions
        int upperBound = ByteVector.SPECIES_64.loopBound(a.length);
        IntVector acc1 = IntVector.zero(IntVector.SPECIES_PREFERRED);
        IntVector acc2 = IntVector.zero(IntVector.SPECIES_PREFERRED);
        for (; i < upperBound; i += ByteVector.SPECIES_64.length()) {
          ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
          ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
          // split each byte vector into two short vectors and multiply
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
