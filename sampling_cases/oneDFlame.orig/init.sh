#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory

cd 0
find . ! -name 'p.orig' ! -name 'T.orig' ! -name 'U.orig' ! -name 'Ydefault.orig' -type f -exec rm -f {} +
# find . ! -name 'p.orig' ! -name 'T.orig' ! -name 'U.orig' ! -name 'Ydefault.orig' -type f -print

cd -

rm -rf dynamicCode processor* 0.* *e-0* log* 2>&1