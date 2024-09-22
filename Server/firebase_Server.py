import firebase_admin
from firebase_admin import credentials, db, storage
import subprocess, time, os

cred = credentials.Certificate('fir-study-e1c26-firebase-adminsdk-jpbew-f354f3515f.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-study-e1c26-default-rtdb.firebaseio.com/',
    'storageBucket': 'fir-study-e1c26.appspot.com'
})


def create_directories():
    # 현재 경로로부터 ../../gaussian-splatting/data에 test2 폴더 생성
    os.makedirs('../../gaussian-splatting/data/test2', exist_ok=True)
    # test2 폴더 안에 input 폴더 생성
    os.makedirs('../../gaussian-splatting/data/test2/input', exist_ok=True)
    print("Create Folder For Train")


def download_images():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="Images/")  # Image 폴더 안의 모든 파일을 가져옴
    for index, blob in enumerate(blobs):
        file_extension = blob.name.split('.')[-1]  # 파일 이름에서 확장자 추출
        destination_filename = f"../../gaussian-splatting/data/test2/input/image{index}.{file_extension}"
        blob.download_to_filename(destination_filename)
        print(f"Downloaded {blob.name} from Firebase Storage as {destination_filename}")

# def download_images():
#     bucket = storage.bucket()
#     for i in range(1, 6):
#         image_blob = bucket.blob(f"image{i}.jpg")
#         image_blob.download_to_filename(f"../../gaussian-splatting/data/test2/input/image{i}.jpg")
#         print("Download images from a Firebase Storage")

	  
def upload_point_cloud():
    bucket = storage.bucket() # Storage 클라이언트 가져오기
    
    # 파일 이름 설정
    storage_file_name = 'point_cloud_test.ply' # Firebase Storage에 저장될 파일 이름
    local_file_path = '../../gaussian-splatting/output/test/point_cloud/iteration_30000/point_cloud.ply'  # 업로드할 로컬 파일 경로 및 이름
	
	# 파일 업로드
    blob = bucket.blob(storage_file_name)
    blob.upload_from_filename(local_file_path)



def monitor_database():
    ref_train = db.reference('/isTrain')
    ref_made = db.reference('/isMade')
    
    while True:
        is_train = ref_train.get()

        if is_train == True:
        	print("#### Gaussian Splatting Operating ####")
             
        	# preprocessing
        	create_directories()
        	download_images()
        	
        	# train + 'test' 라는 인수를 담아서 실행
        	subprocess.run(['./gaussian_splatting.sh','test2'], cwd='../../gaussian-splatting')
        	
        	# 결과 업로드
        	ref_train.set(False)  # is_train 값을 False로 변경
        	upload_point_cloud()
        	ref_made.set(True)  # is_train 값을 False로 변경
        	print("#### Gaussian Splatting Finished ####")
        else:
        	time.sleep(3)  # 3초마다 Realtime Database를 확인
        	print("Waiting Firebase Server--")


if __name__ == "__main__":
    monitor_database()
