# vectorbench

Sorry, you need to install maven, it uses JMH.

## how to run all benchmarks

This may be slow

```console
$ mvn verify
$ java -jar target/vectorbench.jar
```

## listing benchmarks available

```console
$ java -jar target/vectorbench.jar -l
```

## run specific benchmark

```console
$ java -jar target/vectorbench.jar BinaryDotProduct
```

## do the hsdis dance

```console
$ git clone --depth 1 https://github.com/openjdk/jdk/
$ curl https://ftp.gnu.org/gnu/binutils/binutils-2.38.tar.gz | tar -zxf -
$ (cd jdk && bash ./configure --with-hsdis=binutils --with-binutils-src=../binutils-2.38 && make build-hsdis)
$ cp jdk/build/linux-x86_64-server-release/support/hsdis/hsdis-amd64.so $JAVA_HOME/lib/server
```

## get assembler

```console
$ java -jar target/vectorbench.jar -prof perfasm BinaryDotProduct
```

## fake out different vector options

validate on a real machine eventually of course. but dev on a mac is miserable.
You must also set `IS_AMD64_WITHOUT_AVX2 = false` in some benchmarks to test lower sizes on intel.

```console
$ java -jar target/vectorbench.jar -jvmArgsAppend "-XX:MaxVectorSize=16" -prof perfasm BinaryDotProduct
```

## validate correctness (quickly)

This also validates gradle is really running vector tests.
You should see altJvmWarning and should not see gradle say that tests were skipped.

```console
$ export RUNTIME_JAVA_HOME=/home/rmuir/Downloads/jdk-20.0.2+9
$ ./gradlew -p lucene/core test --tests org.apache.lucene.internal.vectorization.TestVectorUtilSupport
```

## before getting excited

```console
$ ./gradlew -p lucene/core test
```
