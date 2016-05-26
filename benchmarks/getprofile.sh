rm profile.png
gprof2dot -f pstats profile.out | dot -Tpng -o profile.png