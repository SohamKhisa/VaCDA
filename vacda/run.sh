# The experiments here will be run for single seed
# For each type of heterogeneity (cross-position, cross-person, cross-device)
# For suitable datasets


experiments=("position" "person" "device")
dataset=("dsads" "opp" "pamap2" "wisdm")
dsads_position=(0 1 2 3 4) # {0:"TORSO", 1:"RA", 2:"LA", 3:"RL", 4:"LL"}
dsads_person=(0 1 2 3 4 5 6 7) # {0:"User1", 1:"User2", 2:"User3", 3:"User4", 4:"User5", 5:"User6", 6:"User7", 7:"User8"}
opp_position=(0 1 2 3 4) # {0:"BACK", 1:"RUA", 2:"RLA", 3:"LUA", 4:"LLA"}
opp_person=(0 1 2 3 4) # {0:"U1", 1:"U2", 2:"U3", 3:"U4"}
pamap_position=(0 1 2) # {0:"Wrist", 1:"Chest", 2:"Ankle"}
pamap_person=(0 1 2 3 4 5 6) # {0:'U1', 1:'U2', 2:'U4', 3:'U5', 4:'U6', 5:'U7', 6:'U8'}
wisdm_person=(0 1 2 3 4 5 6 7 8 9 10) # {0:"U1", 1:"U5", 2:"U7", 3:"U8", 4:"U12", 5:"U13", 6:"U15", 7:"U37", 8:"U38", 9:"U39", 10:"U40"}
wisdm_device=(0 1) # {0:"phone", 1:"watch"}
device="cuda"

# ===================================================
# =============== Cross-position ====================
# ===================================================

# DSADS:
experiment="${experiments[0]}"
for target in "${dsads_position[@]}"; do
  # Source will be all dsads_position elements except the current target
  source=()
  for source_value in "${dsads_position[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[0]}" -s "${source[@]}" -t "$target"
done


# OPPORTUNITY:
experiment="${experiments[0]}"
for target in "${opp_position[@]}"; do
  # Source will be all dsads_position elements except the current target
  source=()
  for source_value in "${opp_position[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[1]}" -s "${source[@]}" -t "$target"
done


# PAMAP2:
experiment="${experiments[0]}"
for target in "${pamap_position[@]}"; do
  # Source will be all dsads_position elements except the current target
  source=()
  for source_value in "${pamap_position[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[2]}" -s "${source[@]}" -t "$target"
done


# ===============================================
# =============== Cross-person ==================
# ===============================================

# DSADS
experiment="${experiments[1]}"
for ((i=${#dsads_person[@]}-1; i>=0; i--)); do
  target="${dsads_person[$i]}"
  source=()
  for source_value in "${dsads_person[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[0]}" -s "${source[@]}" -t "$target"
done


# OPP
experiment="${experiments[1]}"
for ((i=${#opp_person[@]}-1; i>=0; i--)); do
  target="${opp_person[$i]}"
  source=()
  for source_value in "${opp_person[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[1]}" -s "${source[@]}" -t "$target"
done


# WISDM
experiment="${experiments[1]}"
for ((i=${#wisdm_person[@]}-1; i>=0; i--)); do
  target="${wisdm_person[$i]}"
  source=()
  for source_value in "${wisdm_person[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[3]}" -s "${source[@]}" -t "$target"
done




# ===============================================
# =============== Cross-device ==================
# ===============================================

# WISDM
experiment="${experiments[2]}"
for ((i=${#wisdm_device[@]}-1; i>=0; i--)); do
  target="${wisdm_device[$i]}"
  source=()
  for source_value in "${wisdm_device[@]}"; do
    if [ "$source_value" -ne "$target" ]; then
      source+=("$source_value")
    fi
  done
  python vacda.py -g "$device" -e "$experiment" -d "${dataset[3]}" -s "${source[@]}" -t "$target"
done