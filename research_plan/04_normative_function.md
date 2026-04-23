# 4. Normative Function: Delayed commitment는 실제로 planning에 이득을 주는가?

*리뷰어 관점: Correlation (withholding precedes better actions)은 selection bias에 취약하다. 더 좋은 state에서 withhold하기 때문에 좋은 결과가 나올 수 있다. Matched control design이 핵심이다.*

> **섹션 범위 정의**: Section 3가 uncertainty가 NOOP를 *유발하는가*를 물었다면, Section 4는 그 withholding이 실제로 *도움이 되는가*를 묻는다. 분석의 단위는 commit action 이후의 outcome(VRE, k-step reward, 세션 점수)이며, withholding의 기능적 이득을 검증한다. Bout 길이와 outcome의 dose-response 관계도 여기서 다룬다 — 이것은 bout의 *형태*(Section 2)를 기술하는 것이 아니라 bout 길이가 *결과*에 미치는 영향이기 때문이다.

## 4-1. Value Revision Error (VRE)와 k-step reward

### 현재 확보된 지표
- **Value Revision Error (VRE)**: action 선택 시점의 Q와 이후 업데이트된 Q의 차이 → "이 action을 너무 일찍 확신했는가?"
- **k-step reward**: action 이후 k-step 누적 보상합 → "결과적으로 좋았는가?"

### 핵심 질문
- NOOP 직후에 취한 action은 더 낮은 VRE를 가지는가?
- NOOP를 거친 선택은 더 높은 k-step reward를 가지는가?
- Uncertainty가 높은 state일수록 NOOP 후 benefit이 더 큰가?

### 필요한 세부 연구
- NOOP preceding vs. non-NOOP preceding action의 VRE 비교 **[V → Figure 4A]**
- 동일한 uncertainty bin 안에서 NOOP 이후 action의 k-step reward 비교 **[V → Figure 4C]**
- Imitation thinker의 internal tree statistics와 VRE의 관계 **[]**
- "Search conflict가 컸던 state일수록 NOOP 이후 benefit이 큰가" 분석 **[]**

---

## 4-2. Matched-control analysis 강화 *(신규)*

*리뷰어 관점: 현재의 matched-control은 entropy-matching만 한다. 더 엄격한 matching이 필요하다.*

### 핵심 질문
- Entropy뿐 아니라 episode position, game state value, recent reward history를 동시에 matching했을 때도 benefit이 유지되는가?
- Benefit의 effect size는 충분히 크고 일관적인가?
- 더 길게 기다릴수록 더 좋은 결과가 나오는가? (dose-response)

### 필요한 세부 연구
- **Propensity score matching** (withholding 여부를 outcome, uncertainty + episode position + value estimate를 covariates로): matched pair에서 action quality 비교 **[NEW]**
- **Dose-response relationship**: bout length ~ subsequent k-step reward / VRE reduction → 더 길게 기다릴수록 더 좋아지는가? *(Section 2-1에서 이동)* **[NEW]**
- **Interaction: uncertainty × withholding on outcome**: uncertainty가 높은 state에서만 benefit이 명확한가? **[NEW]**

---

## 4-3. Session-level performance benefit *(신규)*

*리뷰어 관점: Trial-level 효과뿐 아니라 session-level 효과가 있어야 "planning이 실제로 기능한다"는 주장이 강해진다.*

### 핵심 질문
- 세션 수준에서 NOOP 비율이 높은 세션이 더 높은 점수를 달성하는가?
- 피험자 수준에서 "더 selective한 withholder"가 더 좋은 performance를 보이는가?

### 필요한 세부 연구
- **Session-level regression**: total score ~ mean NOOP ratio + mean pre-NOOP uncertainty (피험자 random effect 포함) **[NEW]**
- **Selectivity index**: total NOOP count 대비 uncertainty-triggered NOOP 비율이 높은 피험자가 더 잘 하는가? **[NEW]**

---

## 4-4. NOOP ablation (선택적이나 강력한 증거) *(기존 언급 → 구체화)*

반드시 필요하지는 않지만, **있으면 매우 강해진다.**

### 가능한 접근
- **Counterfactual simulation**: imitation thinker를 이용하여, human의 NOOP step에서 만약 immediately action을 취했다면 어떤 결과가 나왔을지 simulate → 실제 결과와 비교
- **NOOP-masked replay**: human trajectory에서 NOOP를 제거하고 thinker가 재연할 때의 성능 비교 **[NEW]**
