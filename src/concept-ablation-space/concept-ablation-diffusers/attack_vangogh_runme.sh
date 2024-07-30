methods=('AC' 'ESD' 'FMN' 'SPM' 'UCE')
# methods=('EraseDiff' )
vangogh_names=('VanGogh')
for vangogh_name in "${vangogh_names[@]}"; do
    for method in "${methods[@]}"; do
        echo "Inverting $method ...."
        python concept-ablation-diffusers/attack_tamed_model_variable_vangogh.py --method $method --prompt $vangogh_name
    done
done

