import os
import numpy as np
import sys
import sqlite3
import random

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def pipeline(scene, base_path, n_views, target_views):
    llffhold = 8
    view_path = str(n_views) + '_'+str(target_views)+'_extra_views'
    os.chdir(base_path + scene)
    os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT')


    images = {}
    with open('../sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                fid.readline().split()
                images[image_name] = elems[1:]

    img_list = sorted(images.keys(), key=lambda x: x)
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
        idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]
        print("original train_img_list====>", train_img_list)
        # Step 1: Determine the test images (every 8th image)
        test_images = [img_list[i] for i in range(len(img_list)) if i % llffhold == 0]

        # Step 2: The remaining images are for the train set
        remaining_images = [img_list[i] for i in range(len(img_list)) if i % llffhold != 0]

        # Step 3: Sample 12 images uniformly spaced from the remaining images
        idx_sub = [round(i) for i in np.linspace(0, len(remaining_images) - 1, n_views)]
        selected_images = [remaining_images[i] for i in idx_sub]
        print("selected_images====>", selected_images)
        # Step 4: Select one image between each pair of selected images
        additional_images = []
        for i in range(len(selected_images) - 1):
            # Get the current gap between two selected images
            start_idx = remaining_images.index(selected_images[i])
            end_idx = remaining_images.index(selected_images[i + 1])

            # Get the images between them and choose one
            in_between = remaining_images[start_idx:end_idx]
            if in_between:
                if target_views == 24:
                    additional_images.append(in_between[len(in_between) // 2])  # Select the middle image
                elif target_views == 48:
                    additional_images.append(in_between[int(len(in_between) * (1/4))])  # Select the middle image
                    additional_images.append(in_between[int(len(in_between) * (2/4))])  # Select the middle image
                    additional_images.append(in_between[int(len(in_between) * (3/4))])  # Select the middle image
        difference = list(set(remaining_images) - set(selected_images + additional_images))
        # chosen_element = random.choice(difference)
        chosen_element = random.sample(difference, 3)
        # Combine the 12 uniformly spaced images and the 12 additional images
        final_train_images = selected_images + additional_images + chosen_element

        # Step 5: Ensure the final train set contains exactly 24 images
        final_train_images = final_train_images[:target_views]
        print("final_train_images====>", len(final_train_images))
    print("train_img_list====>", train_img_list)
    print("train_img_list====>", final_train_images)

    for img_name in final_train_images:
        os.system('cp ../images/' + img_name + '  images/' + img_name)

    os.system('cp ../sparse/0/cameras.txt created/.')
    with open('created/points3D.txt', "w") as fid:
        pass
    
    res = os.popen(
        'colmap feature_extractor --database_path database.db --image_path images --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    os.system(
        'colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    db = COLMAPDatabase.connect('database.db')
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    print(img_rank, res)
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(img_name)
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)
    
    os.system(
        'colmap point_triangulator --database_path database.db --image_path images --input_path created  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    os.system('colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT')
    os.system('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')
    os.system('colmap patch_match_stereo --workspace_path dense')
    os.system('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')
    
    print("Done............")

for scene in ['kitchen']: #['bicycle', 'bonsai', 'counter', 'garden',  'kitchen', 'room', 'stump']:
    pipeline(scene, base_path = '/home/rikhilgupta/Desktop/Data/mipnerf360/', n_views = 12, target_views = 24)  # please use absolute path!