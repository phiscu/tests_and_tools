!#/bin/bash

for field in "t2m-2m_temperature" "tp-total_precipitation" "pev-potential_evaporation" "tev-total_evaporation"; do
#"d2m-2m_dewpoint_temperature" \
#"sf-snowfall" \
#"sp-surface_pressure" \
#"ssrd-surface_solar_radiation_downwards" \
#"strd-surface_thermal_radiation_downwards" \
#"tcc-total_cloud_cover" \
#"u10-10m_u_component_of_wind" \
#"v10-10m_v_component_of_wind" \

    PART=(${field//-/ })
    mkdir "/home/phillip/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/CDC_Download/${PART[0]}"
    for year in `seq 1979 2022`; do
        echo "Downloading field $field year $year"
        sed -e "s/YYYY/$year/g" -e "s/LONG_NAME/${PART[1]}/g" -e "s/NAME/${PART[0]}/g" ERA5_template.py > era5-${PART[0]}.py
        python3 era5-${PART[0]}.py
        done
    done
