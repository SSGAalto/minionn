#Makefile for Miracl, ABY, and SEAL Libraries


#relative project folders for ABY
BIN=bin
SRC=ABY/src
CORE=${SRC}/abycore
ARCHIVE_FILE=libaby.a

# Create bin dir
$(shell  mkdir -p $(BIN))

# all source files and corresponding object files in abycore
SOURCES_CORE := $(shell find ${CORE} -type f -name '*.cpp' -not -path '*/ENCRYPTO_utils/miracl_lib/*' -not -path '*/ENCRYPTO_utils/Miracl/*' -not -path '*/ot/external/*')
OBJECTS_CORE := $(SOURCES_CORE:.cpp=.o)
#OBJECTS_CORE := $(shell find ${CORE} -type f -name '*.o' -not -path '*/ENCRYPTO_utils/miracl_lib/*' -not -path '*/ot/external/*')


# directory for the Miracl submodule and library
MIRACL_LIB_DIR=${CORE}/ENCRYPTO_utils/miracl_lib
SOURCES_MIRACL= ${MIRACL_LIB_DIR}/mrcore.c ${MIRACL_LIB_DIR}/mrarth0.c ${MIRACL_LIB_DIR}/mrarth1.c ${MIRACL_LIB_DIR}/mrarth2.c ${MIRACL_LIB_DIR}/mralloc.c ${MIRACL_LIB_DIR}/mrsmall.c ${MIRACL_LIB_DIR}/mrio1.c ${MIRACL_LIB_DIR}/mrio2.c ${MIRACL_LIB_DIR}/mrgcd.c ${MIRACL_LIB_DIR}/mrjack.c ${MIRACL_LIB_DIR}/mrxgcd.c ${MIRACL_LIB_DIR}/mrarth3.c ${MIRACL_LIB_DIR}/mrbits.c ${MIRACL_LIB_DIR}/mrrand.c ${MIRACL_LIB_DIR}/mrprime.c ${MIRACL_LIB_DIR}/mrcrt.c ${MIRACL_LIB_DIR}/mrscrt.c ${MIRACL_LIB_DIR}/mrmonty.c ${MIRACL_LIB_DIR}/mrpower.c ${MIRACL_LIB_DIR}/mrsroot.c ${MIRACL_LIB_DIR}/mrcurve.c ${MIRACL_LIB_DIR}/mrfast.c ${MIRACL_LIB_DIR}/mrshs.c ${MIRACL_LIB_DIR}/mrshs256.c ${MIRACL_LIB_DIR}/mrshs512.c ${MIRACL_LIB_DIR}/mrsha3.c ${MIRACL_LIB_DIR}/mrfpe.c ${MIRACL_LIB_DIR}/mraes.c ${MIRACL_LIB_DIR}/mrgcm.c ${MIRACL_LIB_DIR}/mrlucas.c ${MIRACL_LIB_DIR}/mrzzn2.c ${MIRACL_LIB_DIR}/mrzzn2b.c ${MIRACL_LIB_DIR}/mrzzn3.c ${MIRACL_LIB_DIR}/mrzzn4.c ${MIRACL_LIB_DIR}/mrecn2.c ${MIRACL_LIB_DIR}/mrstrong.c ${MIRACL_LIB_DIR}/mrbrick.c ${MIRACL_LIB_DIR}/mrebrick.c ${MIRACL_LIB_DIR}/mrec2m.c ${MIRACL_LIB_DIR}/mrgf2m.c ${MIRACL_LIB_DIR}/mrflash.c ${MIRACL_LIB_DIR}/mrfrnd.c ${MIRACL_LIB_DIR}/mrdouble.c ${MIRACL_LIB_DIR}/mrround.c ${MIRACL_LIB_DIR}/mrbuild.c ${MIRACL_LIB_DIR}/mrflsh1.c ${MIRACL_LIB_DIR}/mrpi.c ${MIRACL_LIB_DIR}/mrflsh2.c ${MIRACL_LIB_DIR}/mrflsh3.c ${MIRACL_LIB_DIR}/mrflsh4.c ${MIRACL_LIB_DIR}/mrmuldv.c 
SOURCES_MIRACL_CPP= ${MIRACL_LIB_DIR}/big.cpp ${MIRACL_LIB_DIR}/zzn.cpp ${MIRACL_LIB_DIR}/ecn.cpp ${MIRACL_LIB_DIR}/ec2.cpp ${MIRACL_LIB_DIR}/flash.cpp ${MIRACL_LIB_DIR}/crt.cpp 
OBJECTS_MIRACL=$(SOURCES_MIRACL:.c=.o)
OBJECTS_MIRACL_CPP=$(SOURCES_MIRACL_CPP:.cpp=.o)

OTEXT_DIR=${CORE}/ot
OTEXT_SUB_DIR=${OTEXT_DIR}/external/ot
OT_SUBDIR_FILES=${OTEXT_SUB_DIR}/baseOT.h ${OTEXT_SUB_DIR}/iknp-ot-ext-rec.cpp ${OTEXT_SUB_DIR}/iknp-ot-ext-rec.h ${OTEXT_SUB_DIR}/iknp-ot-ext-snd.cpp ${OTEXT_SUB_DIR}/iknp-ot-ext-snd.h ${OTEXT_SUB_DIR}/naor-pinkas.cpp ${OTEXT_SUB_DIR}/naor-pinkas.h ${OTEXT_SUB_DIR}/ot-ext.cpp ${OTEXT_SUB_DIR}/ot-ext.h ${OTEXT_SUB_DIR}/ot-ext-snd.cpp ${OTEXT_SUB_DIR}/ot-ext-snd.h ${OTEXT_SUB_DIR}/ot-ext-rec.cpp ${OTEXT_SUB_DIR}/ot-ext-rec.h ${OTEXT_SUB_DIR}/xormasking.h ${OTEXT_SUB_DIR}/maskingfunction.h ${OTEXT_SUB_DIR}/kk-ot-ext-snd.h ${OTEXT_SUB_DIR}/kk-ot-ext-snd.cpp ${OTEXT_SUB_DIR}/kk-ot-ext-rec.h ${OTEXT_SUB_DIR}/kk-ot-ext-rec.cpp ${OTEXT_SUB_DIR}/kk-ot-ext.h
OT_FILES=${OTEXT_DIR}/baseOT.h ${OTEXT_DIR}/iknp-ot-ext-rec.cpp ${OTEXT_DIR}/iknp-ot-ext-rec.h ${OTEXT_DIR}/iknp-ot-ext-snd.cpp ${OTEXT_DIR}/iknp-ot-ext-snd.h ${OTEXT_DIR}/naor-pinkas.cpp ${OTEXT_DIR}/naor-pinkas.h ${OTEXT_DIR}/ot-ext.cpp ${OTEXT_DIR}/ot-ext.h ${OTEXT_DIR}/ot-ext-snd.cpp ${OTEXT_DIR}/ot-ext-snd.h ${OTEXT_DIR}/ot-ext-rec.cpp ${OTEXT_DIR}/ot-ext-rec.h ${OTEXT_DIR}/xormasking.h ${OTEXT_DIR}/maskingfunction.h ${OTEXT_DIR}/kk-ot-ext-snd.h ${OTEXT_DIR}/kk-ot-ext-snd.cpp ${OTEXT_DIR}/kk-ot-ext-rec.h ${OTEXT_DIR}/kk-ot-ext-rec.cpp ${OTEXT_DIR}/kk-ot-ext.h


# compiler settings
CC=gcc -fPIC
ABY_COMPILER_OPTIONS=-O2 -std=c++14 -march=native
ABY_COMPILER_OPTIONS_C=-O2 -m64
ARCHITECTURE = $(shell uname -m)
ifeq (${ARCHITECTURE},x86_64)
MIRACL_MAKE:=linux64_cpp
GNU_LIB_PATH:=x86_64
ARCH_LIB_PATH:=64
else
MIRACL_MAKE:=linux
GNU_LIB_PATH:=i386
ARCH_LIB_PATH:=32
endif


LIBRARIES=-lgmp -lgmpxx -lpthread ${CORE}/ENCRYPTO_utils/miracl_lib/miracl.a -L /usr/lib -lssl -lcrypto -lrt



all: miracl miraclbuild otext core ${BIN}/${ARCHIVE_FILE}
	@echo "make all done."

# this will create a copy of the files in src/ENCRYPTO_utils/Miracl and its sub-directories and put them into src/ENCRYPTO_utils/miracl_lib without sub-directories, then compile it
#miracl:	${MIRACL_LIB_DIR}/miracl.a

# copy Miracl files to a new directory (${CORE}/ENCRYPTO_utils/miracl_lib/), call the build script and delete everything except the archive, header and object files.
#${MIRACL_LIB_DIR}/miracl.a:
#	@find ${CORE}/ENCRYPTO_utils/Miracl/ -type f -exec cp '{}' ${CORE}/ENCRYPTO_utils/miracl_lib \;
#	@cp miracl.makefile "${CORE}/ENCRYPTO_utils/miracl_lib/${MIRACL_MAKE}"; cd ${CORE}/ENCRYPTO_utils/miracl_lib/; bash ${MIRACL_MAKE};

miracl: FORCE
	@find ${CORE}/ENCRYPTO_utils/Miracl/ -type f -exec cp '{}' ${MIRACL_LIB_DIR} \;
	@cp ${MIRACL_LIB_DIR}/mirdef.h64 ${MIRACL_LIB_DIR}/mirdef.h
	@cp ${MIRACL_LIB_DIR}/mrmuldv.g64 ${MIRACL_LIB_DIR}/mrmuldv.c

FORCE: ;

miraclbuild: ${OBJECTS_MIRACL} ${OBJECTS_MIRACL_CPP}

# this will create a copy of the files in src/abycore/ot/external/ot/ and put them into src/abycore/ot where they are then used for compiling.
# it does not override files in the ot dir!
otext:
	@cp -n ${OT_SUBDIR_FILES} ${OTEXT_DIR}


core: otext ${OBJECTS_CORE}

%.o:%.cpp %.h
	${CC} $< ${ABY_COMPILER_OPTIONS} -c -o $@

%.o:%.cpp
	${CC} $< ${ABY_COMPILER_OPTIONS} -c -o $@

%.o:%.c
	${CC} $< ${ABY_COMPILER_OPTIONS_C} -c -o $@

${BIN}/${ARCHIVE_FILE}: ${OBJECTS_MIRACL} ${OBJECTS_MIRACL_CPP} ${OBJECTS_CORE}
	ar rc $@ $^

.PHONY: all clean cleanall

clean:
	rm -f ${OBJECTS_CORE}
	rm -f ${OBJECTS_MIRACL}
	rm -f ${MIRACL_LIB_DIR}/miracl.a

cleanall: clean
	rm -f ${BIN}/*.a
	rmdir --ignore-fail-on-non-empty $(BIN)
