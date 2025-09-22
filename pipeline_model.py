import warnings
warnings.filterwarnings('ignore')
import numpy as np
import re, math, pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gensim.models.fasttext import load_facebook_vectors
from pathlib import Path


class PredictionPipeline:
    """완전한 예측 파이프라인 클래스"""
    
    def __init__(self, fasttext_path="./cc.ko.300.bin", model_path="./lightgbm_test_20250919_234611.pkl"):
        """
        예측 파이프라인 초기화
        
        Args:
            fasttext_path (str): FastText 모델 파일 경로
            model_path (str): 저장된 모델 파일 경로
        """
        # FastText 모델 로드
        self.ft_path = Path(fasttext_path)
        if not self.ft_path.exists():
            raise FileNotFoundError(f"fastText .bin 파일이 없습니다: {self.ft_path.resolve()}")
        
        self.ft = load_facebook_vectors(str(self.ft_path))
        self.emb_dim = self.ft.vector_size
        self.UNK = np.zeros(self.emb_dim)  # Unknown 벡터
        
        # 모델 및 전처리기 로드
        # self.model, self.preprocessors, self.feature_names, self.model_params = self._load_complete_model(model_path)
        self.model, self.preprocessors, self.model_params = self._load_complete_model(model_path)

        
        # 파이프라인 프로세서 초기화
        self.processor = self._initialize_processor()
        
        # 등급 및 채널 데이터 설정
        self._setup_mappings()
    
    def _load_complete_model(self, filename):
        """완전한 모델 파이프라인 불러오기"""
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
        
        print(f"모델 불러오기 완료: {filename}")
        print(f"저장 시간: {loaded_dict['timestamp']}")
        
        return (
            loaded_dict['model'],
            loaded_dict['preprocessors'],
            # loaded_dict['feature_names'],
            loaded_dict.get('model_params')
        )
    
    def _initialize_processor(self):
        """최종 파이프라인 프로세서 초기화"""
        return FinalPipelineProcessor(self.model, self.preprocessors, self.ft)
    
    def _setup_mappings(self):
        """등급 및 채널 매핑 데이터 설정"""
        self.c_rank = {'A':1, 'S':2, 'S+':3, 'SS+':6, 'U':7}
        
        self.dict_avg_rank = {
            'A': 151.05988923710987, 
            'S': 201.49720341477774, 
            'S+': 677.2235872235872, 
            'SS+': 4032.4869519832987, 
            'U': 2503.557471264368
        }
        
        self.dict_avg_channel = {
            '강남서초맘': 72.41666666666667, '강남엄마목동엄마': 354.3952291861553, 
            '강북노원도봉맘': 77.93624544981799, '강사모': 158.57399640503294, 
            '검단맘블리': 358.6333333333333, '고고!루원맘스': 230.0, '고고당': 304.91905564924116, 
            '고아캐드': 4664.580927384077, '고양이라서 다행이야': 348.04605263157896, 
            '광주맘스팡': 674.0, '구리남양주맘': 300.75, '구미맘수다방': 677.2235872235872, 
            '구별맘': 339.1105527638191, '금동맘': 164.4645255147717, '김해줌마렐라': 109.18847505270556, 
            '까꿍맘': 108.22207864131438, '나눔카페 [구]광명맘': 332.93781634128703, 
            '나이키매니아': 3097.8382923674, '노원맘': 114.4625784645099, 
            '뉴스사사': 2503.557471264368, '다이렉트 결혼준비': 232.81916243654823, 
            '다이어트는 씨씨앙': 81.61143330571666, '인천아띠아모': 312.8241042345277, 
            '포항맘놀이터': 531.421052631579
        }
        
        self.channel_rank = {
            '뉴스사사': 'U', '나이키매니아': 'SS+', '고아캐드': 'SS+', '구미맘수다방': 'S+',
            '네스프레소 캡슐커피를 사랑하는 모임': 'S', '다이렉트 결혼준비': 'S', 
            '고양이라서 다행이야': 'S', '강사모': 'S', '강남엄마목동엄마': 'A', '구별맘': 'A',
            '검단맘블리': 'A', '광주맘스팡': 'A', '강북노원도봉맘': 'A', '고고!루원맘스': 'A',
            '금동맘': 'A', '김해줌마렐라': 'A', '까꿍맘': 'A', '인천아띠아모': 'A',
            '구리남양주맘': 'A', '나눔카페 [구]광명맘': 'A', '노원맘': 'A', '강남서초맘': 'A',
            '다이어트는 씨씨앙': 'A', '고고당': 'A', '포항맘놀이터': 'A'
        }
    
    def predict_engagement(self, 채널명, p_label, t_label, brand, title_length, 
                          isin_keyword, hour, weekday):
        """
        단일 게시물의 참여도 예측
        
        Args:
            채널명 (str): 채널명
            p_label (str): 제품 라벨
            t_label (str): 게시물 타입 라벨
            brand (str): 브랜드명
            title_length (int): 제목 길이
            isin_keyword (int): 키워드 포함 여부 (0 또는 1)
            hour (int): 시간 (0-23)
            weekday (int): 요일 (0-6)
            
        Returns:
            dict: 예측 결과 및 관련 정보
        """
        try:
            # 등급 정보 가져오기
            등급 = self.channel_rank.get(채널명, 'A')  # 기본값: A
            
            # 예측 실행
            y_pred = self.processor.process_single(
                채널명=채널명,
                p_label=p_label,
                t_label=t_label,
                brand=brand,
                title_length=title_length,
                isin_keyword=isin_keyword,
                hour=hour,
                weekday=weekday
            )
            
            return {
                'success': True,
                'prediction': float(y_pred[0]) if len(y_pred) > 0 else 0.0,
                # 'avg_channel_performance': self.dict_avg_channel.get(채널명, 0),
                # 'avg_grade_performance': self.dict_avg_rank.get(등급, 0),
                # 'input_data': {
                #     '채널명': 채널명,
                #     'p_label': p_label,
                #     't_label': t_label,
                #     'brand': brand,
                #     'title_length': title_length,
                #     'isin_keyword': isin_keyword,
                #     'hour': hour,
                #     'weekday': weekday
                # }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                # 'input_data': {
                #     '채널명': 채널명,
                #     'p_label': p_label,
                #     't_label': t_label,
                #     'brand': brand,
                #     'title_length': title_length,
                #     'isin_keyword': isin_keyword,
                #     'hour': hour,
                #     'weekday': weekday
                # }
            }
    
    def predict_multiple(self, data_list):
        """
        여러 게시물의 참여도 예측
        
        Args:
            data_list (list): 예측할 데이터들의 리스트
                각 항목은 dict 형태: {'채널명': '...', 'p_label': '...', ...}
                
        Returns:
            list: 각 예측 결과들의 리스트
        """
        results = []
        for data in data_list:
            result = self.predict_engagement(**data)
            results.append(result)
        return results
    
    def get_available_channels(self):
        """사용 가능한 채널 목록 반환"""
        return list(self.channel_rank.keys())
    
    def get_channel_info(self, 채널명):
        """특정 채널의 정보 반환"""
        등급 = self.channel_rank.get(채널명)
        if not 등급:
            return {'error': f'채널 {채널명}을 찾을 수 없습니다.'}
        
        return {
            '채널명': 채널명,
            '등급': 등급,
            '등급_수치': self.c_rank.get(등급, 0),
            '평균_채널_성과': self.dict_avg_channel.get(채널명, 0),
            '평균_등급_성과': self.dict_avg_rank.get(등급, 0)
        }


class FinalPipelineProcessor:
    """최종 파이프라인 실시간 처리기"""
    
    def __init__(self, model, package, fasttext_model):
        self.reduced_embeddings = package['reduced_embeddings']
        self.brand_pca = package['brand_pca']
        self.scaler = package['numerical_scaler']
        self.fasttext_model = fasttext_model
        self.model = model
        self.UNK = np.zeros(fasttext_model.vector_size)
    
    def normalize_brand(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        s = re.sub(r"\s+", "", s)
        return s
    
    def get_brand_vector(self, brand: str) -> np.ndarray:
        normalized_brand = self.normalize_brand(brand)
        if not normalized_brand:
            return self.UNK
        try:
            return self.fasttext_model.get_vector(normalized_brand)
        except Exception:
            return self.UNK
    
    def create_time_features(self, hour, day_of_week):
        """시간 순환 인코딩"""
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)
        return np.array([hour_sin, hour_cos, dow_sin, dow_cos])
    
    def process_single(self, 채널명, p_label, t_label, brand, title_length, 
                      isin_keyword, hour, weekday):
        """단일 샘플 실시간 처리"""
        # 매핑 데이터 (클래스 외부에서 전달받거나 여기서 정의)
        c_rank = {'A':1, 'S':2, 'S+':3, 'SS+':6, 'U':7}
        dict_avg_rank = {
            'A': 151.05988923710987, 'S': 201.49720341477774, 
            'S+': 677.2235872235872, 'SS+': 4032.4869519832987, 'U': 2503.557471264368
        }
        dict_avg_channel = {
            '강남서초맘': 72.41666666666667, '강남엄마목동엄마': 354.3952291861553, 
            '강북노원도봉맘': 77.93624544981799, '강사모': 158.57399640503294, 
            '검단맘블리': 358.6333333333333, '고고!루원맘스': 230.0, '고고당': 304.91905564924116, 
            '고아캐드': 4664.580927384077, '고양이라서 다행이야': 348.04605263157896, 
            '광주맘스팡': 674.0, '구리남양주맘': 300.75, '구미맘수다방': 677.2235872235872, 
            '구별맘': 339.1105527638191, '금동맘': 164.4645255147717, '김해줌마렐라': 109.18847505270556, 
            '까꿍맘': 108.22207864131438, '나눔카페 [구]광명맘': 332.93781634128703, 
            '나이키매니아': 3097.8382923674, '노원맘': 114.4625784645099, 
            '뉴스사사': 2503.557471264368, '다이렉트 결혼준비': 232.81916243654823, 
            '다이어트는 씨씨앙': 81.61143330571666, '인천아띠아모': 312.8241042345277, 
            '포항맘놀이터': 531.421052631579
        }
        channel_rank = {
            '뉴스사사': 'U', '나이키매니아': 'SS+', '고아캐드': 'SS+', '구미맘수다방': 'S+',
            '네스프레소 캡슐커피를 사랑하는 모임': 'S', '다이렉트 결혼준비': 'S', 
            '고양이라서 다행이야': 'S', '강사모': 'S', '강남엄마목동엄마': 'A', '구별맘': 'A',
            '검단맘블리': 'A', '광주맘스팡': 'A', '강북노원도봉맘': 'A', '고고!루원맘스': 'A',
            '금동맘': 'A', '김해줌마렐라': 'A', '까꿍맘': 'A', '인천아띠아모': 'A',
            '구리남양주맘': 'A', '나눔카페 [구]광명맘': 'A', '노원맘': 'A', '강남서초맘': 'A',
            '다이어트는 씨씨앙': 'A', '고고당': 'A', '포항맘놀이터': 'A'
        }
        
        # 1. 카테고리 임베딩
        cafe_emb = self.reduced_embeddings["채널명"][채널명]
        product_emb = self.reduced_embeddings["p_label"][p_label]
        post_emb = self.reduced_embeddings["t_label"][t_label]
        cat_embeddings = np.hstack([cafe_emb, product_emb, post_emb])
        
        # 2. 브랜드
        brand_vector = self.get_brand_vector(brand)
        brand_features = self.brand_pca.transform(brand_vector.reshape(1, -1))[0]
        
        # 3. 수치형
        등급 = channel_rank[채널명]
        avg_rank = dict_avg_rank[등급]
        avg_channel = dict_avg_channel[채널명]
        numerical_raw = np.array([[title_length, avg_rank, avg_channel]])  
        numerical_features = self.scaler.transform(numerical_raw).flatten()
        
        # 4. 이진 + 등급
        등급_수치 = c_rank[등급]
        binary_features = np.array([isin_keyword, 등급_수치])      
        
        # 5. 시간
        time_features = self.create_time_features(hour, weekday)
        
        # 6. 최종 결합
        features_list = [cat_embeddings, brand_features, numerical_features, binary_features, time_features]        
        final_features = np.hstack(features_list)

        input_data = final_features.reshape(1, -1)
        y_pred_log = self.model.predict(input_data)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred


# 사용 예시
if __name__ == "__main__":
    # 파이프라인 초기화
    pipeline = PredictionPipeline()
    
    # 단일 예측
    result = pipeline.predict_engagement(
        채널명='나이키매니아',
        p_label='스포츠',
        t_label='정보공유(정리/팁)',
        brand="스타벅스",
        title_length=45,
        isin_keyword=1,
        hour=14,
        weekday=1
    )
    
    print("단일 예측 결과:")
    print(f"성공: {result['success']}")
    if result['success']:
        print(f"예측값: {result['prediction']:.2f}")
        # print(f"채널 등급: {result['channel_grade']}")
    else:
        print(f"오류: {result['error']}")
    
    # 여러 예측
    data_list = [
        {
            '채널명': '나이키매니아',
            'p_label': '스포츠',
            't_label': '정보공유(정리/팁)',
            'brand': '나이키',
            'title_length': 30,
            'isin_keyword': 1,
            'hour': 10,
            'weekday': 0
        },
        {
            '채널명': '강남서초맘',
            'p_label': '패션',
            't_label': '딜/프로모션(핫딜·할인·쿠폰·증정)',
            'brand': '자라',
            'title_length': 25,
            'isin_keyword': 0,
            'hour': 15,
            'weekday': 3
        }
    ]
    
    results = pipeline.predict_multiple(data_list)
    print(f"\n다중 예측 결과: {len(results)}개")
    for i, result in enumerate(results):
        if result['success']:
            print(f"{i+1}번째: {result['prediction']:.2f}")
        else:
            print(f"{i+1}번째: 오류 - {result['error']}")
    
    # 사용 가능한 채널 목록
    channels = pipeline.get_available_channels()
    print(f"\n사용 가능한 채널 수: {len(channels)}")
    
    # 특정 채널 정보
    channel_info = pipeline.get_channel_info('나이키매니아')
    print(f"\n나이키매니아 정보: {channel_info}")