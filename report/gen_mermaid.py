import base64
import json

mermaid_code = """graph TD
    subgraph "Part 1: Pre-Processing"
    A([Start]) --> B[Load Image & Grayscale]
    B --> C[Laplacian Filter & Threshold]
    C --> D[Find External Contours]
    D --> E[Create Solid Gear Mask]
    E --> F[Distance Transform & Root Circle]
    end

    subgraph "Part 2: Defect Detection"
    G[Subtract Body Mask for Gear Teeth] --> H[Morphological Opening]
    H --> I[Detect Tooth Contours]
    I --> J[Sort Areas & Calculate Trimmed Average]
    J --> K[Compare with Trimmed Avg for Defect]
    K --> L[Render Output Images & Print Results]
    L --> M([End])
    end
    
    F -.-> G
"""

payload = {"code": mermaid_code, "mermaid": {"theme": "default"}}
json_str = json.dumps(payload)
b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')

print(f"https://mermaid.ink/img/{b64_str}")
