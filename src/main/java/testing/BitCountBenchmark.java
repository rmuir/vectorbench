package testing;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import jdk.incubator.vector.LongVector;
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
public class BitCountBenchmark {

  private long[] longs;

  @Param({"1024"})
  int size;

  @Setup(Level.Trial)
  public void init() {
    longs = new long[size];
    for (int i = 0; i < size; i++) {
      longs[i] = ThreadLocalRandom.current().nextLong();
    }
  }

  @Benchmark
  public int bitCountOld() {
    int sum = 0;
    for (int i = 0; i < size; i++) {
      sum += Long.bitCount(longs[i]);
    }
    return sum;
  }

  static final VectorSpecies<Long> PREFERRED_LONG_SPECIES = LongVector.SPECIES_PREFERRED;

  @Benchmark
  public int bitCountNew() {
    int i = 0;
    int res = 0;
    int upperBound = PREFERRED_LONG_SPECIES.loopBound(longs.length);
    LongVector acc = LongVector.zero(PREFERRED_LONG_SPECIES);
    for (; i < upperBound; i += PREFERRED_LONG_SPECIES.length()) {
      LongVector longVector = LongVector.fromArray(PREFERRED_LONG_SPECIES, longs, i);
      LongVector bitCount = longVector.lanewise(VectorOperators.BIT_COUNT);
      acc = acc.add(bitCount);
    }
    res += (int) acc.reduceLanes(VectorOperators.ADD);
    for (; i < longs.length; i++) {
      res += Long.bitCount(longs[i]);
    }
    return res;
  }
}
