from openai import OpenAI
import os, json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import pandas as pd
from tqdm.auto import tqdm
import re
import ast


class LLMClassifier:
    def __init__(self, env_path='.env', summary_infos_path='summary_infos2.json', 
                 summary_dict_kor_path='summary_dict_kor.json', TYPE_TO_SECTIONS_path='TYPE_TO_SECTIONS.json'):
        """
        LLM을 이용한 텍스트 분류 및 생성 클래스
        
        Args:
            env_path (str): .env 파일 경로
            summary_infos_path (str): summary_infos2.json 파일 경로
            summary_dict_kor_path (str): summary_dict_kor.json 파일 경로
            TYPE_TO_SECTIONS_path (str): TYPE_TO_SECTIONS.json 파일 경로
        """
        # 환경변수 로드
        dotenv_path = os.path.join(os.getcwd(), '.', env_path)
        load_dotenv(dotenv_path)
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=os.getenv("GPT_API_KEY"))
        
        self.path = './jsons/'
        # JSON 파일들 로드
        with open(self.path + summary_infos_path, 'r', encoding='utf-8') as f:
            self.summary_infos = json.load(f)
        with open(self.path + summary_dict_kor_path, 'r', encoding='utf-8') as f:
            self.summary_dict_kor = json.load(f)
        with open(self.path + TYPE_TO_SECTIONS_path, 'r', encoding='utf-8') as f:
            self.TYPE_TO_SECTIONS = json.load(f)
        
        # 카페 목록
        self.cafes = ['fashion', 'health', 'etc', 'mom']
        
        # 기본 콘텐츠 타입
        self.DEFAULT_TYPE = '딜/프로모션(핫딜·할인·쿠폰·증정)'
    

    def get_emoticon_guide(self, cafe_tone_json):
        """이모티콘 사용 가이드 생성"""
        emoticon_ratio = float(cafe_tone_json.get('이모티콘 비율', '0%').replace('%', ''))
        if emoticon_ratio >= 20:
            return "이모티콘을 자주 사용하세요 (😊, 👍, 💕, 🔥 등을 문장 사이사이 적절히 배치)"
        elif emoticon_ratio >= 15:
            return "이모티콘을 종종 사용하세요 (😊, 👍 정도를 2-3번 포함)"
        elif emoticon_ratio >= 10:
            return "이모티콘을 가끔 사용하세요 (😊 또는 👍 을 1-2번 포함)"
        elif emoticon_ratio >= 5:
            return "이모티콘을 적게 사용하세요 (😊 정도를 1번만 포함)"
        else:
            return "이모티콘을 사용하지 마세요"
        
    """브랜드 추출용 프롬프트 생성"""
    def prompt1(self, title, summary_f_toks):
        message = f"""
        아래 글의 "대표브랜드(b)"와 "제목 내포 여부(f)"를 추출해 JSON만 반환하라.
        반환 형식: {{"b": <str>, "f": <int>}}
        규칙:
        - 우선순위: 제목 > 요약토큰.
        - 여러 후보면 가장 명시적 판매/행사 주체 또는 최다 빈도 1개.
        - 모델명/제품군/행사명 제외, 브랜드(회사/스토어명)만.
        - "로켓배송/스마트스토어/라이브커머스/쿠폰" 등 플랫폼/기능 제외.
        - 온라인 줄임말/영문 약칭을 한글 풀네임으로 표기(예: LG→엘지, KB증권→케이비증권).
        - 브랜드를 못 찾는 경우:
        - 이벤트성 게시물이면 행사 주최사를 b로 반환.
        - 그래도 불가하면 "미정".
        - f: 대표브랜드 판단을 제목으로 했다면 1, 아니면 0.
        - 모든 반환은 한글. 배열([]) 금지, 코드펜스/여분 텍스트 금지, 설명 금지. JSON 1개만.

        제목: {title}
        요약토큰: {summary_f_toks}
        """
        return message.strip()


    """콘텐츠 생성용 프롬프트 생성"""
    def prompt2(self, product, cafe_tone_json, content_type_str):
        picked_type = self.DEFAULT_TYPE if content_type_str=='데이터X' else content_type_str
        sections = self.TYPE_TO_SECTIONS.get(picked_type, self.TYPE_TO_SECTIONS[self.DEFAULT_TYPE])
        emoticon_guide = self.get_emoticon_guide(cafe_tone_json)
        message = f"""
        반드시 JSON만 출력하고, 키는 정확히 "title"과 "content" 두 개만 사용하라.
        마크다운/코드블록/불릿/번호/섹션명(예: 훅, 장점, 방법, 주의, CTA 등) 및 어떤 라벨도 출력하지 마라.
        일반 문장들로만 자연스럽게 작성하라.

        [입력]
        - 상품명: {product}
        - 카페 톤 지표(JSON): {json.dumps(cafe_tone_json, ensure_ascii=False)}
        - 선택된 본문 유형: {picked_type}

        [톤 규칙]
        - 해요/합니다: "해요체 비율" > "합니다체 비율"이면 해요체, 아니면 합니다체.
        - 조건·할인율 같은 숫자는 나열하지 말고, 문장 속에 자연스럽게 녹여서 설명한다.
        - 단순 정보 나열 대신, 실제 사용 경험담이나 감정 표현(편하다, 좋았다, 아쉽다 등)을 섞어준다.

        [이모티콘 및 표현 규칙]
        - {emoticon_guide} 
        - 이모티콘 뒤에는 문장보호 '.' 을 넣지 않는다.

        [내용 구성 가이드]
        - {sections}

        [제목 규칙]
        - 25자 내외, 중복구두점 금지, 상품 키워드 1개 포함 권장.

        [출력 형식]
        {{"title":"<문자열>","content":"<문자열>"}}
        """
        return message.strip()
            
        
    def classify(self, prompt_text):
        """통합 분류 실행 - 브랜드 추출과 콘텐츠 생성 모두 처리"""
        r = self.client.responses.create(
            model="gpt-5-nano",
            input=prompt_text,
            reasoning={"effort":"low"},
            text={"verbosity": "low"}
        )
        txt = r.output_text
        if not txt:
            try:
                txt = r.output[0].content[0].text
            except Exception:
                for part in getattr(r.output[0], "content", []):
                    if getattr(part, "type", "") == "text":
                        txt = part.text
                        break
        return json.loads(txt)
    

    def process_brand_extraction(self, title, summary_f_toks):
        """
        브랜드 추출 프로세스
        
        Args:
            title (str): 제목
            summary_f_toks (str): 요약 토큰들
            
        Returns:
            dict: 브랜드 추출 결과
        """
        prompt_text = self.prompt1(title, summary_f_toks)
        result = self.classify(prompt_text)
        
        return {
            # 'title': title,
            # 'summary_f_toks': summary_f_toks,
            # 'brand_result': result,
            'brand': result.get('b', '미정'),
            # 'title_based': result.get('f', 0)
        }
    
    def process_content_generation(self, product, ptype):
        """
        콘텐츠 생성 프로세스 (모든 카페 타입에 대해)
        
        Args:
            product (str): 상품명
            ptype (str): 상품 유형
            
        Returns:
            dict: 모든 카페 타입별 생성 결과
        """
        results = {
            'product': product,
            'ptype': ptype,
            'cafe_results': {}
        }
        
        for ctype in self.cafes:
            try:
                prompt_text = self.prompt2(
                    product, 
                    self.summary_dict_kor[ctype], 
                    self.summary_infos[ctype][ptype]
                )
                answer = self.classify(prompt_text)
                
                results['cafe_results'][ctype] = {
                    'success': True,
                    # 'result': answer,
                    'title': answer.get('title', ''),
                    'summary': answer.get('content', ''),
                }
            except Exception as e:
                results['cafe_results'][ctype] = {
                    'success': False,
                    'error': str(e),

                }
        
        return results

# 사용 예시
if __name__ == "__main__":
    # 클래스 인스턴스 생성
    classifier = LLMClassifier()
    
    # 브랜드 추출만 실행
    title = '이건 제목입니다'
    summary_f_toks = '본문'
    brand_result = classifier.process_brand_extraction(title, summary_f_toks)
    print("Brand extraction result:", brand_result)
    
    # 콘텐츠 생성만 실행
    product = '테스트 상품'
    ptype = '패션'
    content_result = classifier.process_content_generation(product, ptype)
    print("Content generation result:", content_result)
    