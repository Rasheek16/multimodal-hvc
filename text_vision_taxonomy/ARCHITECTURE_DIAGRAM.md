# Taxonomy Moderation Architecture Diagram

This diagram shows the current recommended production architecture. The taxonomy labels come from the text transformer model. The text and vision base models provide hate/severity gating and calibration signals.

```mermaid
flowchart TD
    A[Raw video input] --> B[Frame sampler]
    A --> C[ASR transcript]
    B --> D[OCR text]
    B --> E[Vision base model<br/>frames -> hate + severity]

    C --> F[Metadata builder]
    D --> F
    G[Manual metadata optional<br/>Scene / Action / Subcategories / ParentLabels / action_clean] --> F

    F --> H[Taxonomy text input builder]
    H --> I[RoBERTa text taxonomy classifier<br/>multi-label sigmoid outputs]

    I --> J[Per-label probabilities]
    J --> K[Calibrated production thresholds]
    K --> L[Rare-label rules<br/>threat / illegal / online_harm]

    C --> M[Text base model<br/>text -> hate + severity]
    D --> M
    F --> M

    E --> N[Base model signals]
    M --> N
    N --> O[Base-signal gating]
    L --> O

    O --> P[Final taxonomy decision]
    P --> Q[Final JSON output]

    Q --> Q1[metadata]
    Q --> Q2[text hate/severity]
    Q --> Q3[vision hate/severity]
    Q --> Q4[taxonomy_raw_labels]
    Q --> Q5[taxonomy_final_labels]
    Q --> Q6[candidate_labels]
    Q --> Q7[suppressed_labels]
    Q --> Q8[evidence]

    R[Experimental video taxonomy head] -. disabled for production now .-> O
    B -. frames .-> R

    classDef main fill:#e8f1ff,stroke:#2563eb,stroke-width:1px,color:#111827;
    classDef model fill:#eef9f0,stroke:#16a34a,stroke-width:1px,color:#111827;
    classDef guard fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#111827;
    classDef output fill:#f8fafc,stroke:#475569,stroke-width:1px,color:#111827;
    classDef disabled fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray:4 4,color:#374151;

    class A,B,C,D,F,G,H main;
    class E,I,M model;
    class K,L,N,O guard;
    class P,Q,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8 output;
    class R disabled;
```

## Training And Calibration Flow

```mermaid
flowchart TD
    A[final_dataset.csv<br/>1060 rows] --> B[Source-video split by File Name]
    B --> C[Train split<br/>750 rows]
    B --> D[Validation split<br/>156 rows]
    B --> E[Test split<br/>154 rows]

    C --> F[Build taxonomy text<br/>Scene + Action + Subcategories + ParentLabels + action_clean]
    F --> G[RoBERTa multi-label training<br/>9 sigmoid labels]

    G --> H[Validation probabilities]
    D --> H
    H --> I[Threshold calibration]
    I --> I1[max F1 thresholds]
    I --> I2[precision target thresholds]
    I --> I3[top-k prevalence thresholds]
    I --> I4[rare-label conservative floors]
    I4 --> J[Production thresholds]

    G --> K[Test probabilities]
    E --> K
    J --> L[Test evaluation]

    L --> M[Current calibrated test result<br/>micro precision 0.9290<br/>micro recall 0.8276<br/>micro F1 0.8754<br/>rare FP 0]

    N[Runtime feedback logs] --> O[Review queue / pseudo labels]
    O --> P[Feedback retraining dataset<br/>1060 original + 1 pseudo row]
    P --> G

    classDef data fill:#e8f1ff,stroke:#2563eb,stroke-width:1px,color:#111827;
    classDef model fill:#eef9f0,stroke:#16a34a,stroke-width:1px,color:#111827;
    classDef eval fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#111827;
    classDef result fill:#f8fafc,stroke:#475569,stroke-width:1px,color:#111827;

    class A,B,C,D,E,F,N,O,P data;
    class G model;
    class H,I,I1,I2,I3,I4,J,K,L eval;
    class M result;
```

## Current Production Recommendation

```text
Use:
  RoBERTa text taxonomy classifier
  + calibrated thresholds
  + rare-label rules
  + text hate/severity base model
  + vision hate/severity base model

Do not use as final taxonomy source yet:
  video taxonomy head

Reason:
  video taxonomy currently has incomplete frame coverage and unstable rare-label behavior.
```

