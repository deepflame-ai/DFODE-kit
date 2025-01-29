#!/bin/sh
cd 0_1DFlameTemplate/ && pwd
rm -rf ./*.yaml
cd - && echo ""

cd 1_dataGeneration/ && pwd
rm -rf ./*
touch .gitkeep
cd - && echo ""

cd 2_sampling/ && pwd
./clean.sh
cd - && echo ""

cd 4_validation/0_1DFlame/ && pwd
rm -rf *_1DFlame_*
cd - && echo ""
