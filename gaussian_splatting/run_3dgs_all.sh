python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/fern -m output/llff_3dgs_new/fern --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/fern -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/fern

python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/flower -m output/llff_3dgs_new/flower --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/flower -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/flower

python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/fortress -m output/llff_3dgs_new/fortress --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/fortress -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/fortress

python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/horns -m output/llff_3dgs_new/horns --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/horns -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/horns

python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/leaves -m output/llff_3dgs_new/leaves --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/leaves -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/leaves

python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/orchids -m output/llff_3dgs_new/orchids --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/orchids -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/orchids


python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/room -m output/llff_3dgs_new/room --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/room -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/room

python train.py -s /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/data/nerf_llff_data/trex -m output/llff_3dgs_new/trex --eval -r 4
python render.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/trex -r 4 --iteration 7000
python metrics.py -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs_new/trex
