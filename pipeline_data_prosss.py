import re, joblib, json
from soynlp.tokenizer import MaxScoreTokenizer
from konlpy.tag import Mecab
from soynlp.normalizer import repeat_normalize


class TextProcessor:
    def __init__(self, mecab_dicpath='C:/mecab/mecab-ko-dic', scores_path='./jsons/combined_scores_ts.joblib'
                 ,keywords_path = './jsons/keywords_config.json'):
        """
        텍스트 전처리 및 토크나이징을 위한 클래스
        
        Args:
            mecab_dicpath (str): MeCab 사전 경로
            scores_path (str): joblib 저장된 combined_scores 파일 경로
        """
        self.m = Mecab(dicpath=mecab_dicpath)

        with open(keywords_path, 'r', encoding='utf-8') as f:
            self.keyword_dict = json.load(f)
        
        # 정규표현식 패턴들 정의
        self._setup_regex_patterns()
        
        # 품사 태깅 설정
        self._setup_pos_settings()
        
        # combined_scores 로딩 및 토크나이저 초기화
        self.combined_scores = joblib.load(scores_path)
        self.tokenizer = MaxScoreTokenizer(scores=self.combined_scores)
        
        # 토큰 정제용 정규표현식
        self.PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)
        self.SPLIT_INNER = re.compile(r"[^\w가-힣]+", flags=re.UNICODE)
    
    def _setup_regex_patterns(self):
        """정규표현식 패턴들을 설정합니다."""
        # 1) 전화번호
        self.RE_TEL = re.compile(r"""
        (?<!\d)(
            0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}      |  # 02/0xx/010-(1234)-(5678)
            0\d{1,2}\d{7,8}                        |  # 0212345678, 01012345678
            \(\s?0\d{1,2}\s?\)[-\s]?\d{3,4}[-\s]?\d{4} |  # (02) 1234-5678
            \+82[-\s]?0?\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4} |  # +82-10-1234-5678
            1[5-9]\d{2}[-\s]?\d{4}                 |  # 1588-1234 류 대표번호
            0\d{1,2}[-\s]?\*{3,4}[-\s]?\d{4}          # 010-****-1234
        )(?!\d)
        """, re.X)

        # 2) URL
        self.RE_URL = re.compile(
            r"(?i)\b("
            r"(?:https?://|ftp://|www\.)\S+|"
            r"(?:open\.kakao\.com/[^\s]+)|"
            r"(?:kakaolink://[^\s]+)|"
            r"(?:naver\.me/[^\s]+)|"
            r"(?:me2\.do/[^\s]+)|"
            r"(?:forms\.gle/[^\s]+)|"
            r"(?:linktr\.ee/[^\s]+)|"
            r"(?:bit\.ly/[^\s]+)|"
            r"(?:t\.co/[^\s]+)|"
            r"(?:smartstore\.naver\.com/[^\s]+)|"
            r"(?:shopping\.naver\.com/[^\s]+)|"
            r"(?:brand\.naver\.com/[^\s]+)|"
            r"(?:cafe\.naver\.com/[^\s]+)|"
            r"(?:m\.cafe\.naver\.com/[^\s]+)"
            r")"
        )

        # 3) 이메일
        self.RE_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")

        # 4) 가격
        self.RE_PRICE = re.compile(r"""
        (?:
            ₩\s*[\d,]+                         |   # ₩39,900
            [\d,]+\s*원                        |   # 39,900원
            \d+(?:\.\d+)?\s*만(?:\s*원)?       |   # 5만 / 5만 원 / 5.5만(원)
            \d+\s*만\s*\d+\s*천\s*(?:\d{2,3})?\s*원? | # 3만9천(900)원, 12만 5천원 등
            \d+\s*천\s*(?:\d{2,3})?\s*원?      |   # 9천(900)원
            만원                                   # '만원' 단독
        )
        """, re.X)

        # 5) 퍼센트
        self.RE_PCT = re.compile(r"\b\d{1,3}\s?(?:%|퍼|프로)\b")

        # 6) 날짜/시간
        self.RE_DATE = re.compile(r"\b(?:\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[/.-]\d{1,2})\b")
        self.RE_TIME = re.compile(r"\b(?:오전|오후)?\s?\d{1,2}:\d{2}\b")

        # 7) 해시태그/멘션
        self.RE_HASHTAG = re.compile(r"#[\w가-힣]+")
        self.RE_MENTION = re.compile(r"@\w+")

        # 8) 이모지
        self.RE_EMOJI = re.compile(
            "[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF"
            "\U00002700-\U000027BF\U00002600-\U000026FF]+", flags=re.UNICODE
        )
    
    def _setup_pos_settings(self):
        """품사 태깅 관련 설정을 합니다."""
        self.ALLOW_PREFIX = ("NN",)
        self.ALLOW_EXACT = {"SL", "SH"}
        
        self.PUNCT = {"SF", "SE", "SSO", "SSC", "SC", "SY"}
        self.BAN_EXACT = {"MAG", "MAJ", "IC"} | self.PUNCT
        self.BAN_PREFIX = ("J", "E", "X")
    
    def normalize(self, text: str) -> str:
        """텍스트 정규화를 수행합니다."""
        t = text or ""
        t = re.sub(r'[\r\n]+', ' ', t)
        
        # 반복 문자 정규화
        t = repeat_normalize(t, num_repeats=2)
        
        # 정규표현식 치환
        t = self.RE_TEL.sub("<TEL>", t)
        t = self.RE_URL.sub("<URL>", t)
        t = self.RE_EMAIL.sub("<EMAIL>", t)
        t = self.RE_PRICE.sub("<PRICE>", t)
        t = self.RE_PCT.sub("<PCT>", t)
        t = self.RE_DATE.sub("<DATE>", t)
        t = self.RE_TIME.sub("<TIME>", t)
        t = self.RE_HASHTAG.sub("<HASHTAG>", t)
        t = self.RE_MENTION.sub("<MENTION>", t)
        t = self.RE_EMOJI.sub("<EMOJI>", t)
        
        # 공백 정리
        t = re.sub(r"\s+", " ", t).strip()
        return t
        
    def keep_token(self, token: str) -> bool:
        """토큰을 유지할지 판단합니다."""
        if not token or token.strip() == "":
            return False
        try:
            pairs = self.m.pos(token)
        except Exception:
            return True
        
        """명사군 품사인지 확인합니다."""
        return any(tag.startswith(self.ALLOW_PREFIX) or (tag in self.ALLOW_EXACT) for _, tag in pairs)
    
    def clean_tokens(self, tokens):
        """토큰들을 필터링하고 정제합니다."""
        # 필터링: keep_token과 길이 조건 적용
        filtered_tokens = [t for t in tokens if self.keep_token(t) and len(t) >= 2]
        
        out = []
        for t in filtered_tokens:
            if not t:
                continue
            # 내부 기호 기준 분할
            parts = [p for p in self.SPLIT_INNER.split(str(t)) if p]
            for p in parts:
                p = self.PUNCT_EDGE.sub("", p).strip()
                if p:
                    out.append(p)
        
        # 중복 제거(원래 순서 유지)
        seen, cleaned = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                cleaned.append(p)
        return cleaned
    
    def check_keyword(self, p_label, tokens):
        """
        토큰들이 해당 제품 라벨의 키워드에 포함되는지 확인
        
        Args:
            p_label (str): 제품 라벨 (예: '패션', '뷰티' 등)
            tokens (list): 확인할 토큰 리스트
            
        Returns:
            int: 키워드 포함 시 1, 미포함 시 -1
        """
        if p_label not in self.keyword_dict:
            return -1
        dictionary = self.keyword_dict[p_label]
        
        for token in tokens:
            if token in dictionary:
                return 1
        
        return -1
    
    def process_text(self, p_label, title, summary, hour):
        """
        전체 텍스트 처리 파이프라인을 실행합니다.
        
        Args:
            p_label (str): 패션 라벨
            t_label (str): 질문 라벨  
            title (str): 제목
            summary (str): 본문
            hour (int): 시간
            
        Returns:
            dict: 모든 처리 결과를 담은 딕셔너리
        """
        # 입력값들
        original_title = title
        original_summary = summary
        
        # 텍스트 정규화
        normalized_title = self.normalize(title)
        normalized_summary = self.normalize(summary)
        
        # 길이 계산
        t_length = len(normalized_title)
        s_length = len(normalized_summary)
        
        # 토크나이징
        t_toks = self.tokenizer.tokenize(normalized_title or "", flatten=True)
        s_toks = self.tokenizer.tokenize(normalized_summary or "", flatten=True)

        # 토큰 정제
        t_f_toks = self.clean_tokens(t_toks)
        s_f_toks = self.clean_tokens(s_toks)

        isin_k =self.check_keyword(p_label, t_f_toks)
        
        # 결과 딕셔너리 생성
        results = {
            # 입력값들
            'p_label': p_label,
            'original_title': original_title,
            # 'original_summary': original_summary,
            'hour': hour,
            
            # 정규화된 텍스트
            # 'normalized_title': normalized_title,
            # 'normalized_summary': normalized_summary,
            
            # 길이
            't_length': t_length,
            's_length': s_length,
            
            # 토크나이징 결과
            # 't_toks': t_toks,
            't_f_toks': t_f_toks,
            's_f_toks': s_f_toks,

            'isin_keyword': isin_k
            
            # # 클래스 내부 객체들
            # 'combined_scores': self.combined_scores,
            # 'tokenizer': self.tokenizer,
            # 'mecab': self.m
        }
        
        return results


# 사용 예시
if __name__ == "__main__":
    # 클래스 인스턴스 생성
    processor = TextProcessor()
    
    # 입력 데이터
    p_label = '패션'
    t_label = '질문 (Q&A)'
    title = '이건 제목입니다.'
    summary = '이건 본문입니다.'
    hour = 7
    
    # 텍스트 처리 실행
    results = processor.process_text(p_label, title, summary, hour)
    
    # 결과 출력
    print("Processing Results:")
    print(f"Original title: {results['original_title']}")
    print(f"Filtered tokens: {results['t_f_toks']}")
    print(f"Filtered tokens: {results['s_f_toks']}")
    print(f"Title length: {results['t_length']}")
    print(f"Summary length: {results['s_length']}")
    print(f"isin_keyword: {results['isin_keyword']}")