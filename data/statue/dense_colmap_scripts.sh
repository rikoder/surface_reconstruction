
# colmap feature_extractor --database_path database.db --image_path images  --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
# colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768
# colmap mapper \
#     --database_path database.db \
#     --image_path images \
#     --output_path sparse/

colmap point_triangulator --database_path database.db --image_path images --input_path sparse/0  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100
colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT
colmap image_undistorter --image_path images --input_path triangulated --output_path dense
colmap patch_match_stereo --workspace_path dense
colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply
