# 김민석의 포트폴리오
(학사) 고려대학교 국어국문학과      
(석사) 서울대학교 언어학과(컴퓨터 언어학 전공)
***           

#### 🍊 **INTRO**                

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 안녕하세요! NLP 엔지니어 **김민석**입니다.             
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 도메인의 **언어적 특성**을 다룰 수 있는 모델을 만듭니다.                        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **금융, 의료, 법률 등 다양한 도메인**을 위한 모델을 개발한 경험이 있습니다. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  


#### 🍊 **PROJECTS**
- [1. (표현의 자유를 보장하는) **비속어 탐지 모델 개발**](#1-표현의-자유를-보장하는-비속어-탐지-모델-개발)
- [2. (한/영 의료 용어를 인식하는) **의료 분야 검색 모델 개발**](#2-한영-의료-용어를-인식하는-의료-분야-검색-모델-개발)
- [3. (RAG 파이프라인을 이용한) **검색 기반 한국어 LLM 개발**](#3-rag-파이프라인을-이용한-검색-기반-한국어-llm-개발)
- [4. (교통사고 과실을 계산하는) **웹기반 한국어 LLM 개발 참여**](#4-교통사고-과실을-계산하는-웹기반-한국어-llm-개발-참여)       
- [5. (다양한 언어를 처리하는) **금융/경제 분야 문장 임베딩 모델 개발**](#5-다양한-언어를-처리하는-금융경제-분야-문장-임베딩-모델-개발)
- [6. (4가지 방법론을 통한) **뉴스 기사 토픽 분류 모델 개발**](#6-4가지-방법론을-통한-뉴스-기사-토픽-분류-모델-개발) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                

## 1. (표현의 자유를 보장하는) 비속어 탐지 모델 개발
> 개인 프로젝트      
> 언어: Python         
> 기술 스택: Pytorch, Django          
> 코드:          
> 블로그:            

<img src="gif/hatespeech.gif" width="450" height="250" alt="Hate Speech Detection (Demo)">

유저의 비속어를 필터링하는 모델입니다.          
표현의 자유를 보장하기 위해 감탄사 또는 자신을 향한 비속어는 필터링되지 않도록 모델링했습니다.

**(1) 문제 의식**
- 서비스 이용자의 유형에 따라 **비속어 필터링의 기준**이 다를 수 있습니다.  
- 예를 들어, 게임에서는 **감탄사**(예: 아 ㅁㅊ)나 **자신을 향한 비속어**(예: 내가 ㅄ)는 관대하게 처리할 수 있습니다.    

**(2) 모델 및 데이터 선택**
- 유저의 채팅을 신속하게 처리하기 위해 **CNN + GRU 구조**의 모델을 선택했습니다.  
- SmileGate, BEEP! 등의 오픈 데이터를 사용했고, **검수**를 통해 위 사례에 해당하는 샘플의 레이블을 수정했습니다.   

**(3) 초성 및 특수문자 비속어 처리**
- 초성을 이용한 비속어 처리를 위해 **초성 단위**로 모델을 학습했습니다.  
- 특수문자를 어휘사전의 각 초성에 맵핑하는 기능을 추가해 특수문자 비속어를 처리했습니다. (예: ㉦→ㅅ) 

**(4) 테스트용 웹 개발**
- Django를 이용해 직접 채팅을 입력해볼 수 있는 테스트용 웹을 구현했습니다.
- 대량의 채팅을 실시간으로 처리하는 상황을 가정하고 **비동기 프로그래밍**을 적용했습니다.
            
**(5) 문제 해결**     

- 데이터를  레이블을 수정해 
- 

## 2. (한/영 의료 용어를 인식하는) 의료 분야 검색 모델 개발  
> 개인 프로젝트, 석사논문      
> 언어: Python         
> 기술 스택: Pytorch, Django          
> 코드:          
> 블로그: 

## 3. (RAG 파이프라인을 이용한) 검색 기반 한국어 LLM 개발  
> 교통사고 과실 상계 분야            
>

## 4. (교통사고 과실을 계산하는) 웹기반 한국어 LLM 개발 참여
> 교통사고 과실 상계 분야            
>

## 5. (다양한 언어를 처리하는) 금융/경제 분야 문장 임베딩 모델 개발



## 6. (4가지 방법론을 통한) 뉴스 기사 토픽 분류 모델 개발 
> 개인 프로젝트            
> 언어: Python           
> 기술 스택: Pytorch             
> 코드:          
> 블로그:                        

총 네 가지 방법으로 Sequence Classification을 수행했습니다.                    
일반적인 Classification 방법 외에도 MLM, Matching, Seq2Seq 방식을 통해 분류 문제를 해결했습니다.

__(1) 문제 의식__

<img src="img/classification.PNG" width="450" height="250" alt="Classification">

- Classifier를 이용하는 일반적인 Sequence Classification은 **대량의 학습 데이터를** 필요로 합니다. 
- BERT 모델에 기반한 방법론이므로 최근 발전하는 **디코더 모델**에 그대로 적용할 수 없습니다.

__(2) 대안 ① : Masked Language Modeling__

<img src="img/mlm.PNG" width="450" height="250" alt="Masked Language Modeling">

- **Pattern Exploiting Training(PET)** 을 통해 MLM 문제를 Classification 문제로 전환했습니다. 
  - "(문장)의 주제는 \[MASK\]이다" 등의 **패턴**에서 \[MASK\]에 대한 모든 토큰의 확률을 계산했습니다.   
  - "생활 → 생활/문화" 처럼 토큰을 레이블에 연결하는 **Verbalizer**를 이용해 각 레이블의 확률을 종합했습니다.
- 다양한 패턴과 Verbalizer를 실험하며 **최적의 성능**을 내는 조합을 탐색했습니다.
- **사전학습과 유사한 튜닝 방식(MLM)** 을 사용함으로써 동일한 크기의 데이터로 분류 성능을 향상시켰습니다.  

__(3) 대안 ② : Matching__

<img src="img/entailment.PNG" width="450" height="250" alt="Entailment">

- 템플릿을 이용해 Multi-class Classification 문제를 **Binary Classification** 문제로 전환했습니다. 
  - "(문장) \[SEP\] 이 문장의 주제는 (레이블)이다"의 구성에서 참일 확률이 가장 높은 레이블을 찾는 방식입니다.   
  - 동일한 문장에 대해 각 레이블의 확률을 비교해야하므로 한 문장에서 비롯된 구성들을 묶어서 처리했습니다.
- 다양한 템플릿을 실험하며 **최적의 성능**을 내는 템플릿을 탐색했습니다.
- **사전학습과 유사한 튜닝 방식(NSP)** 을 사용함으로써 동일한 크기의 데이터로 분류 성능을 향상시켰습니다.  


__(4) 대안 ④ : Seq2Seq__

<img src="img/seq2seq.PNG" width="450" height="250" alt="Seq2Seq">

- BART 모델에서 제시한 방법대로 **디코더의 표현**을 이용해 Classification을 수행했습니다. 
  - 디코더는 다음 토큰을 예측하는 방식으로 학습하므로 **마지막 토큰의 표현**을 사용했습니다.   
  - 마지막 토큰의 정보까지 온전히 활용하기 위해 문장의 끝에 **특수 토큰**을 추가했습니다.
- **디코더 모델**에 적용할 수 있는 방법으로 Classification을 수행했습니다.
