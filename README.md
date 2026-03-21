```
python generate_data_csv.py --index_file ../GEMS_data/v2020/index/INDEX_general_PL_data.2020 --split_json ./data/CleanSplit/PDBbind_data_split_cleansplit.json --split_name casf2016 --output casf16.csv
```
python dataprep.py --index_file ./example_dataset/example_dataset.csv --work_dir example_dataset
python dataset.py --index_file example_dataset/example_dataset.csv --work_dir example_dataset/ --output example_dataset/example.pt

python test.py --config ./config/config_temporalsplit_gine.json --model_paths '["./checkpoints/temporalsplit_gine/best_full_gine_temporalsplit_s44.pt"]'

python test.py --config ./config/config_cleansplit_gine.json --variant "full" --num_layers "4" --model_paths '["./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f0.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f1.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f2.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f3.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f4.pt"]'

python test.py --config ./config/config_pdbbind2020_gine.json --model_paths '["./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f0.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f1.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f2.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f3.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f4.pt"]'

python inference.py --config ./config/config_cleansplit_gine.json --protein_pdb data/example_dataset/1a1e/1a1e_protein.pdb --ligand_mol2 data/example_dataset/1a1e/1a1e_ligand.mol2 --model_paths '["./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f0.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f1.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f2.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f3.pt","./checkpoints/pdbbind2020_gine/best_full_gine_pdbbind2020_f4.pt"]'

python residue_attribution.py --config ./config/config_cleansplit_gine.json --graph_pt_path ./data_predealed/residue_attr/casf16.pt --save_dir ./residue_attr_casf16 --model_paths '["./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f0.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f1.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f2.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f3.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f4.pt"]'

bash analysis/run_gbsa.sh /Path/to/CASF16

python analysis/summary.py --model_attr_dir residue_attr_casf16/ --work_dir data_predealed/CASF16_min/ --out_dir ./GBSA_casf16

python analysis/spearman.py --attr_dir residue_attr_casf16/ --energy_dir GBSA_casf16   --energy_suffix pocket_decomp   --out_csv spearman_per_complex_casf16.csv

python analysis/accuracy_recall.py   --attr_dir residue_attr_casf16/   --energy_dir GBSA_casf16/   --out_csv accuracy_recall.csv