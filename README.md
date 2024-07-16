# 김민석의 포트폴리오

__🍊 안녕하세요! NLP 엔지니어 김민석입니다.__             
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __도메인의 특성을 이해하는 것을 중요하게 생각합니다.__                        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __금융, 의료, 법률, 게임 분야의 텍스트 처리를 위한 모델을 개발한 경험이 있습니다.__ 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  

- [1. (표현의 자유를 보장하는) 비속어 탐지 모델 개발](#1-표현의-자유를-보장하는-비속어-탐지-모델-개발)
- [2. (한/영 의료 용어를 인식하는) 의료 분야 검색 모델 개발](#2-한-영-의료-용어를-인식하는-의료-분야-검색-모델-개발)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  

## 1. (표현의 자유를 보장하는) 비속어 탐지 모델 개발
> 개인 프로젝트      
> 언어: Python         
> 기술 스택: Pytorch, Django         

유저의 비속어를 필터링하는 모델입니다.          
표현의 자유를 최대한 보장하기 위해 감탄사 또는 자신을 향한 비속어는 필터링되지 않도록 모델링했습니다.

__모델 선택__
- 비속어 탐지의 특성상 모델의 크기보다 데이터의 양과 품질이 중요  
- 유저의 채팅을 신속하게 처리하기 위해 CNN + GRU 구조 선택  

__초성/특수문자 비속어 처리__
- 초성을 이용한 비속어 처리를 위해 초성 단위로 모델 학습  
- 학습 데이터의 일부 초성을 유사한 형태의 특수 문자로 대체해 특수문자까지 처리

__테스트용 웹 개발__
- Django를 이용해 직접 채팅을 입력해볼 수 있는 테스트용 웹 구현
- 대량의 채팅을 실시간으로 처리하기 위해 비동기 프로그래밍 사용

__트러블 슈팅__
- __문제__
  - 감탄사나 자신을 향한 비속어(ex. ㅅㅂ, ㅁㅊ, 아 내가 바보임)까지 필터링 되는 문제 발생
  - 학습 데이터가 욕설을 포함하면 모두 비속어로 분류된 데에서 비롯된 문제
  - 게임 등 정제되지 않은 언어가 용인되는 분야에서 표현의 자유를 침해할 여지가 있음
- __해결__
  - 전이 학습을 거친 모델을 위와 같은 유형을 포함한 소규모 데이터로 Fine-tuning 하여 문제 완화
  - Sentence Pair의 관계를 추론하는 방식으로 전이 학습과 Fine-tuning 을 진행해 둘 사이의 간격을 좁힘
  - ex) Sentence Pair : "아 내가 바보임", "이 문장은 '비속어' 이다.", Label: False         

## 2. (한/영 의료 용어를 인식하는) 의료 분야 검색 모델 개발  

## 3. 웹기반 한국어 LLM 개발 참여
> 교통사고 과실 상계 분야            
>

## 4. RAG 기반 검색 엔진 개발
> 교통사고 과실 상계 분야            
>

## 5. 다양한 방법론을 이용한 토픽 분류 
> 개인 프로젝트            
> 언어: Python           
> 기술 스택: Pytorch          

총 네 가지 방법으로 Topic Classification을 수행합니다. 
다양한 

__Classification__

__Masked Language Modeling__

__Matching__

__Seq2Seq__


## 6. 금융/경제 분야 다국어 문장 임베딩 모델 개발
