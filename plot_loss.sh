./parse_log.sh $1
gnuplot -e "filename='$1.test'" plot_log.gnuplot
