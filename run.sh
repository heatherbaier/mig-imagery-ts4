ISO="MEX4"
ADM="2"
IC="LANDSAT/LT05/C01/T1"

python3 save_boxes_ipums.py $ISO
python3 calc_centroids.py $ISO
python3 download_imagery.py $IC --year_list "2010" --month_list "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12"
