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
