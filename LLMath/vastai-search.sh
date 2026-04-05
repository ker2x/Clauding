vastai search offers 'gpu_ram >= 8 reliability > 0.9 inet_down > 100' -o 'dlperf_per_dphtotal-' --type ondemand --raw | \
  jq -r '["ID","GPU","VRAM","$/hr","DLperf","DLP/$","up$/TB","dn$/TB"],
          (sort_by(-.dlperf_per_dphtotal) | .[] | [.id, "\(.num_gpus)x \(.gpu_name)", (.gpu_ram|round),
                  (.dph_total * 100 | round | . / 100),
                  (.dlperf | round),
                  (.dlperf_per_dphtotal | round),
                  (.inet_up_cost * 1000 * 100 | round | . / 100),
                  (.inet_down_cost * 1000 * 100 | round | . / 100)])
          | map(tostring) | join("|")' | column -t -s '|'
