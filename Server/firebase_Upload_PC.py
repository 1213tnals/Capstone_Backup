import firebase_admin
from firebase_admin import credentials, storage

# Firebase 프로젝트의 서비스 계정 키를 포함하는 JSON 파일의 경로
cred = credentials.Certificate('fir-study-e1c26-firebase-adminsdk-jpbew-f354f3515f.json')

# Firebase 앱 초기화
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fir-study-e1c26.appspot.com'
})

# Storage 클라이언트 가져오기
bucket = storage.bucket()

# 업로드할 로컬 파일 경로
local_file_path = 'point_cloud.ply'

# Firebase Storage에 저장될 파일 이름
storage_file_name = 'point_cloud3.ply'

# 업로드 실행
blob = bucket.blob(storage_file_name)
blob.upload_from_filename(local_file_path)

print(f'File {local_file_path} uploaded to Firebase Storage as {storage_file_name}')

