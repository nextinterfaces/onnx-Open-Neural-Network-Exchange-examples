# Architecture Diagram

```mermaid
graph TB
    subgraph Client["Client (client.py)"]
        C1[Test Cases] --> C2[HTTP Request]
        C2 --> C3[Response Processing]
        C3 --> C4[Results Display]
    end

    subgraph FastAPI_Server["FastAPI Server (app.py)"]
        F1["/compute_similarity/ Endpoint"] --> F2[Input Validation]
        F2 --> F3[Tokenization]
        F3 --> F4[ONNX Inference]
        F4 --> F5[Mean Pooling]
        F5 --> F6[Cosine Similarity]
        F6 --> F7[JSON Response]
    end

    subgraph Models["Models & Resources"]
        M1[ONNX Model]
        M2[Tokenizer]
    end

    subgraph Data_Types["Data Types (Pydantic)"]
        D1[SentencesInput]
        D2[SimilarityResponse]
    end

    C2 -->|HTTP POST| F1
    F7 -->|JSON| C3
    F3 -->|Use| M2
    F4 -->|Use| M1
    F1 -->|Validate| D1
    F7 -->|Format| D2

    classDef server fill:#f9f,stroke:#333,stroke-width:2px
    classDef client fill:#bbf,stroke:#333,stroke-width:2px
    classDef model fill:#bfb,stroke:#333,stroke-width:2px
    classDef datatype fill:#fbb,stroke:#333,stroke-width:2px

    class F1,F2,F3,F4,F5,F6,F7 server
    class C1,C2,C3,C4 client
    class M1,M2 model
    class D1,D2 datatype
```

## Flow Description

1. **Client Side (client.py)**
   - Prepares test cases with different sentences
   - Sends HTTP POST requests to the server
   - Processes and displays the response

2. **Server Side (app.py)**
   - FastAPI endpoint receives POST request
   - Validates input using Pydantic models
   - Processes sentences through the pipeline:
     * Tokenization
     * ONNX model inference
     * Mean pooling
     * Cosine similarity calculation
   - Returns JSON response

3. **Models & Resources**
   - ONNX model for sentence embeddings
   - Tokenizer for text preprocessing

4. **Data Types**
   - SentencesInput: Validates incoming requests
   - SimilarityResponse: Structures the response

## Technical Details

- **Server Port**: 8000
- **Endpoint**: /compute_similarity/
- **Request Format**: JSON with list of sentences
- **Response Format**: JSON with embeddings and similarities
- **Model Location**: Parent directory (../langcache-embed-v1-model.onnx) 