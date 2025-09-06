# 🔧 API Usage Examples

## 📋 Table of Contents

1. [cURL Examples](#curl-examples)
2. [Python Examples](#python-examples)
3. [JavaScript Examples](#javascript-examples)
4. [Java Examples](#java-examples)
5. [C# Examples](#c-examples)
6. [PHP Examples](#php-examples)
7. [Postman Collection](#postman-collection)

## 🌐 cURL Examples

### Basic Health Check
```bash
curl -X GET "http://localhost:8000/health" \
     -H "Accept: application/json"
```

### Prediction with Standard Model
```bash
curl -X POST "http://localhost:8000/pneumonia/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

### Prediction with EfficientNet Model
```bash
curl -X POST "http://localhost:8000/pneumonia/predict?model=efficientnet_b0" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

### Get Model Information
```bash
curl -X GET "http://localhost:8000/pneumonia/model/info?model=standard" \
     -H "Accept: application/json"
```

### Security Status
```bash
curl -X GET "http://localhost:8000/security/status" \
     -H "Accept: application/json"
```

### Security Statistics
```bash
curl -X GET "http://localhost:8000/security/stats" \
     -H "Accept: application/json"
```

## 🐍 Python Examples

### Using requests library

```python
import requests
import json
from pathlib import Path

class PneumoniaDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self):
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def predict_pneumonia(self, image_path, model="standard"):
        """
        Predict pneumonia from chest X-ray image
        
        Args:
            image_path (str): Path to chest X-ray image
            model (str): Model to use ('standard' or 'efficientnet_b0')
        
        Returns:
            dict: Prediction results or None if failed
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return None
            
        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': (image_path.name, image_file, 'image/jpeg')}
                params = {'model': model}
                
                response = self.session.post(
                    f"{self.base_url}/pneumonia/predict",
                    files=files,
                    params=params,
                    timeout=30
                )
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Prediction failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"HTTP {e.response.status_code}: {e.response.text}")
            return None
    
    def get_model_info(self, model="standard"):
        """Get information about the AI model"""
        try:
            params = {'model': model}
            response = self.session.get(
                f"{self.base_url}/pneumonia/model/info",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return None
    
    def get_security_status(self):
        """Get security system status"""
        try:
            response = self.session.get(f"{self.base_url}/security/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get security status: {e}")
            return None

# Example usage
if __name__ == "__main__":
    client = PneumoniaDetectionClient()
    
    # Check API health
    print("🏥 Checking API health...")
    health = client.health_check()
    if health:
        print(f"✅ API Status: {health['status']}")
        print(f"🤖 Model Loaded: {health['model_loaded']}")
        print(f"📌 Version: {health['version']}")
    
    # Make prediction
    print("\n🔬 Making pneumonia prediction...")
    image_path = "chest_xray.jpg"  # Replace with your image path
    
    result = client.predict_pneumonia(image_path, model="efficientnet_b0")
    if result:
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"💡 Recommendation: {result['medical_recommendation']}")
        print(f"🤖 Model: {result['model_type']}")
    
    # Get model information
    print("\n📊 Getting model information...")
    model_info = client.get_model_info("efficientnet_b0")
    if model_info:
        print(f"🧠 Architecture: {model_info.get('architecture', 'N/A')}")
        print(f"⚡ Inference Time: {model_info.get('inference_time_ms', 'N/A')}ms")
        print(f"🎯 Accuracy: {model_info.get('validation_accuracy', 'N/A')}")
```

### Async Python Example

```python
import aiohttp
import asyncio
from pathlib import Path

class AsyncPneumoniaDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def predict_pneumonia(self, image_path, model="standard"):
        """Async prediction with aiohttp"""
        image_path = Path(image_path)
        
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as image_file:
                data = aiohttp.FormData()
                data.add_field('file', image_file, 
                              filename=image_path.name, 
                              content_type='image/jpeg')
                
                params = {'model': model}
                
                async with session.post(
                    f"{self.base_url}/pneumonia/predict",
                    data=data,
                    params=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error = await response.text()
                        print(f"Error {response.status}: {error}")
                        return None

# Example usage
async def main():
    client = AsyncPneumoniaDetectionClient()
    result = await client.predict_pneumonia("chest_xray.jpg")
    if result:
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")

# Run async example
# asyncio.run(main())
```

## 🌐 JavaScript Examples

### Node.js with axios

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

class PneumoniaDetectionClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 30000
        });
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            console.error('Health check failed:', error.message);
            return null;
        }
    }

    async predictPneumonia(imagePath, model = 'standard') {
        try {
            if (!fs.existsSync(imagePath)) {
                throw new Error(`Image file not found: ${imagePath}`);
            }

            const form = new FormData();
            form.append('file', fs.createReadStream(imagePath));

            const response = await this.client.post('/pneumonia/predict', form, {
                headers: form.getHeaders(),
                params: { model }
            });

            return response.data;
        } catch (error) {
            console.error('Prediction failed:', error.response?.data || error.message);
            return null;
        }
    }

    async getModelInfo(model = 'standard') {
        try {
            const response = await this.client.get('/pneumonia/model/info', {
                params: { model }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to get model info:', error.message);
            return null;
        }
    }

    async getSecurityStatus() {
        try {
            const response = await this.client.get('/security/status');
            return response.data;
        } catch (error) {
            console.error('Failed to get security status:', error.message);
            return null;
        }
    }
}

// Example usage
async function main() {
    const client = new PneumoniaDetectionClient();

    // Check health
    console.log('🏥 Checking API health...');
    const health = await client.healthCheck();
    if (health) {
        console.log(`✅ Status: ${health.status}`);
        console.log(`🤖 Model Loaded: ${health.model_loaded}`);
    }

    // Make prediction
    console.log('\n🔬 Making prediction...');
    const imagePath = './chest_xray.jpg'; // Replace with your image path
    const result = await client.predictPneumonia(imagePath, 'efficientnet_b0');
    
    if (result) {
        console.log(`🎯 Prediction: ${result.prediction}`);
        console.log(`📊 Confidence: ${result.confidence.toFixed(3)}`);
        console.log(`💡 Recommendation: ${result.medical_recommendation}`);
    }
}

main().catch(console.error);
```

### Browser JavaScript (Fetch API)

```javascript
class PneumoniaDetectionWebClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }

    async predictPneumonia(file, model = 'standard') {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const url = new URL(`${this.baseURL}/pneumonia/predict`);
            url.searchParams.append('model', model);

            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Prediction failed:', error);
            return null;
        }
    }

    async getModelInfo(model = 'standard') {
        try {
            const url = new URL(`${this.baseURL}/pneumonia/model/info`);
            url.searchParams.append('model', model);

            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get model info:', error);
            return null;
        }
    }
}

// Example HTML integration
function setupFileUpload() {
    const client = new PneumoniaDetectionWebClient();
    const fileInput = document.getElementById('chest-xray-input');
    const resultDiv = document.getElementById('result');

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        resultDiv.innerHTML = '🔄 Analyzing image...';

        const result = await client.predictPneumonia(file, 'efficientnet_b0');
        
        if (result) {
            resultDiv.innerHTML = `
                <h3>🔬 Analysis Results</h3>
                <p><strong>Prediction:</strong> ${result.prediction}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                <p><strong>Recommendation:</strong> ${result.medical_recommendation}</p>
                <p><strong>Model:</strong> ${result.model_type}</p>
                <p><em>${result.disclaimer}</em></p>
            `;
        } else {
            resultDiv.innerHTML = '❌ Analysis failed. Please try again.';
        }
    });
}

// Call this when the page loads
// setupFileUpload();
```

## ☕ Java Examples

```java
import java.io.*;
import java.net.http.*;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import java.util.Map;

public class PneumoniaDetectionClient {
    private final String baseURL;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public PneumoniaDetectionClient(String baseURL) {
        this.baseURL = baseURL;
        this.httpClient = HttpClient.newHttpClient();
        this.objectMapper = new ObjectMapper();
    }

    public Map<String, Object> healthCheck() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseURL + "/health"))
            .GET()
            .build();

        HttpResponse<String> response = httpClient.send(request, 
            HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return objectMapper.readValue(response.body(), 
                new TypeReference<Map<String, Object>>() {});
        } else {
            throw new RuntimeException("Health check failed: " + response.statusCode());
        }
    }

    public Map<String, Object> predictPneumonia(String imagePath, String model) 
            throws Exception {
        Path path = Paths.get(imagePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Image file not found: " + imagePath);
        }

        // Create multipart form data
        String boundary = "----WebKitFormBoundary" + System.currentTimeMillis();
        
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(outputStream, "UTF-8"));

        // File part
        writer.append("--").append(boundary).append("\r\n");
        writer.append("Content-Disposition: form-data; name=\"file\"; filename=\"")
              .append(path.getFileName().toString()).append("\"\r\n");
        writer.append("Content-Type: image/jpeg\r\n\r\n");
        writer.flush();

        outputStream.write(Files.readAllBytes(path));

        writer.append("\r\n");
        writer.append("--").append(boundary).append("--\r\n");
        writer.flush();

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseURL + "/pneumonia/predict?model=" + model))
            .header("Content-Type", "multipart/form-data; boundary=" + boundary)
            .POST(HttpRequest.BodyPublishers.ofByteArray(outputStream.toByteArray()))
            .build();

        HttpResponse<String> response = httpClient.send(request, 
            HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return objectMapper.readValue(response.body(), 
                new TypeReference<Map<String, Object>>() {});
        } else {
            throw new RuntimeException("Prediction failed: " + response.statusCode() + 
                " - " + response.body());
        }
    }

    public Map<String, Object> getModelInfo(String model) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseURL + "/pneumonia/model/info?model=" + model))
            .GET()
            .build();

        HttpResponse<String> response = httpClient.send(request, 
            HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return objectMapper.readValue(response.body(), 
                new TypeReference<Map<String, Object>>() {});
        } else {
            throw new RuntimeException("Failed to get model info: " + response.statusCode());
        }
    }

    public static void main(String[] args) {
        try {
            PneumoniaDetectionClient client = 
                new PneumoniaDetectionClient("http://localhost:8000");

            // Health check
            System.out.println("🏥 Checking API health...");
            Map<String, Object> health = client.healthCheck();
            System.out.println("✅ Status: " + health.get("status"));
            System.out.println("🤖 Model Loaded: " + health.get("model_loaded"));

            // Make prediction
            System.out.println("\n🔬 Making prediction...");
            Map<String, Object> result = client.predictPneumonia(
                "chest_xray.jpg", "efficientnet_b0");
            
            System.out.println("🎯 Prediction: " + result.get("prediction"));
            System.out.println("📊 Confidence: " + result.get("confidence"));
            System.out.println("💡 Recommendation: " + result.get("medical_recommendation"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## 🔷 C# Examples

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.Json;
using System.Collections.Generic;

public class PneumoniaDetectionClient
{
    private readonly HttpClient httpClient;
    private readonly string baseURL;

    public PneumoniaDetectionClient(string baseURL = "http://localhost:8000")
    {
        this.baseURL = baseURL;
        this.httpClient = new HttpClient();
        this.httpClient.Timeout = TimeSpan.FromSeconds(30);
    }

    public async Task<Dictionary<string, object>> HealthCheckAsync()
    {
        try
        {
            var response = await httpClient.GetAsync($"{baseURL}/health");
            response.EnsureSuccessStatusCode();
            
            var jsonString = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<Dictionary<string, object>>(jsonString);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Health check failed: {ex.Message}");
            return null;
        }
    }

    public async Task<Dictionary<string, object>> PredictPneumoniaAsync(
        string imagePath, string model = "standard")
    {
        try
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"Image file not found: {imagePath}");
            }

            using var form = new MultipartFormDataContent();
            var fileContent = new ByteArrayContent(await File.ReadAllBytesAsync(imagePath));
            fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
            form.Add(fileContent, "file", Path.GetFileName(imagePath));

            var response = await httpClient.PostAsync(
                $"{baseURL}/pneumonia/predict?model={model}", form);
            
            response.EnsureSuccessStatusCode();
            
            var jsonString = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<Dictionary<string, object>>(jsonString);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Prediction failed: {ex.Message}");
            return null;
        }
    }

    public async Task<Dictionary<string, object>> GetModelInfoAsync(string model = "standard")
    {
        try
        {
            var response = await httpClient.GetAsync(
                $"{baseURL}/pneumonia/model/info?model={model}");
            response.EnsureSuccessStatusCode();
            
            var jsonString = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<Dictionary<string, object>>(jsonString);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to get model info: {ex.Message}");
            return null;
        }
    }

    public void Dispose()
    {
        httpClient?.Dispose();
    }
}

// Example usage
class Program
{
    static async Task Main(string[] args)
    {
        var client = new PneumoniaDetectionClient();

        try
        {
            // Health check
            Console.WriteLine("🏥 Checking API health...");
            var health = await client.HealthCheckAsync();
            if (health != null)
            {
                Console.WriteLine($"✅ Status: {health["status"]}");
                Console.WriteLine($"🤖 Model Loaded: {health["model_loaded"]}");
            }

            // Make prediction
            Console.WriteLine("\n🔬 Making prediction...");
            var result = await client.PredictPneumoniaAsync("chest_xray.jpg", "efficientnet_b0");
            if (result != null)
            {
                Console.WriteLine($"🎯 Prediction: {result["prediction"]}");
                Console.WriteLine($"📊 Confidence: {result["confidence"]}");
                Console.WriteLine($"💡 Recommendation: {result["medical_recommendation"]}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        finally
        {
            client.Dispose();
        }
    }
}
```

## 🐘 PHP Examples

```php
<?php

class PneumoniaDetectionClient {
    private $baseURL;

    public function __construct($baseURL = 'http://localhost:8000') {
        $this->baseURL = $baseURL;
    }

    public function healthCheck() {
        $curl = curl_init();
        
        curl_setopt_array($curl, [
            CURLOPT_URL => $this->baseURL . '/health',
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ['Accept: application/json'],
            CURLOPT_TIMEOUT => 10
        ]);

        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        curl_close($curl);

        if ($httpCode === 200) {
            return json_decode($response, true);
        } else {
            error_log("Health check failed: HTTP $httpCode");
            return null;
        }
    }

    public function predictPneumonia($imagePath, $model = 'standard') {
        if (!file_exists($imagePath)) {
            error_log("Image file not found: $imagePath");
            return null;
        }

        $curl = curl_init();
        
        $postFields = [
            'file' => new CURLFile($imagePath, 'image/jpeg', basename($imagePath))
        ];

        curl_setopt_array($curl, [
            CURLOPT_URL => $this->baseURL . "/pneumonia/predict?model=$model",
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => $postFields,
            CURLOPT_TIMEOUT => 30
        ]);

        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        $error = curl_error($curl);
        curl_close($curl);

        if ($error) {
            error_log("cURL error: $error");
            return null;
        }

        if ($httpCode === 200) {
            return json_decode($response, true);
        } else {
            error_log("Prediction failed: HTTP $httpCode - $response");
            return null;
        }
    }

    public function getModelInfo($model = 'standard') {
        $curl = curl_init();
        
        curl_setopt_array($curl, [
            CURLOPT_URL => $this->baseURL . "/pneumonia/model/info?model=$model",
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ['Accept: application/json'],
            CURLOPT_TIMEOUT => 10
        ]);

        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        curl_close($curl);

        if ($httpCode === 200) {
            return json_decode($response, true);
        } else {
            error_log("Failed to get model info: HTTP $httpCode");
            return null;
        }
    }

    public function getSecurityStatus() {
        $curl = curl_init();
        
        curl_setopt_array($curl, [
            CURLOPT_URL => $this->baseURL . '/security/status',
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ['Accept: application/json'],
            CURLOPT_TIMEOUT => 10
        ]);

        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        curl_close($curl);

        if ($httpCode === 200) {
            return json_decode($response, true);
        } else {
            error_log("Failed to get security status: HTTP $httpCode");
            return null;
        }
    }
}

// Example usage
$client = new PneumoniaDetectionClient();

// Health check
echo "🏥 Checking API health...\n";
$health = $client->healthCheck();
if ($health) {
    echo "✅ Status: " . $health['status'] . "\n";
    echo "🤖 Model Loaded: " . ($health['model_loaded'] ? 'true' : 'false') . "\n";
}

// Make prediction
echo "\n🔬 Making prediction...\n";
$imagePath = 'chest_xray.jpg'; // Replace with your image path
$result = $client->predictPneumonia($imagePath, 'efficientnet_b0');

if ($result) {
    echo "🎯 Prediction: " . $result['prediction'] . "\n";
    echo "📊 Confidence: " . number_format($result['confidence'], 3) . "\n";
    echo "💡 Recommendation: " . $result['medical_recommendation'] . "\n";
    echo "🤖 Model: " . $result['model_type'] . "\n";
}

// Get model info
echo "\n📊 Getting model information...\n";
$modelInfo = $client->getModelInfo('efficientnet_b0');
if ($modelInfo) {
    echo "🧠 Architecture: " . ($modelInfo['architecture'] ?? 'N/A') . "\n";
    echo "⚡ Inference Time: " . ($modelInfo['inference_time_ms'] ?? 'N/A') . "ms\n";
    echo "🎯 Accuracy: " . ($modelInfo['validation_accuracy'] ?? 'N/A') . "\n";
}

?>
```

## 📮 Postman Collection

Create a Postman collection with these requests:

### 1. Health Check
```
GET {{base_url}}/health
```

### 2. Pneumonia Prediction
```
POST {{base_url}}/pneumonia/predict?model=standard
Body: form-data
Key: file (File)
Value: [Select chest X-ray image]
```

### 3. Model Information
```
GET {{base_url}}/pneumonia/model/info?model=efficientnet_b0
```

### 4. Security Status
```
GET {{base_url}}/security/status
```

### 5. Security Statistics
```
GET {{base_url}}/security/stats
```

### Environment Variables
```
base_url: http://localhost:8000
```

### Pre-request Scripts
```javascript
// Set timestamp for logging
pm.globals.set("timestamp", new Date().toISOString());
```

### Tests Scripts (for health check)
```javascript
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has correct structure", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('status');
    pm.expect(jsonData).to.have.property('model_loaded');
    pm.expect(jsonData).to.have.property('version');
});

pm.test("Service is healthy", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData.status).to.be.oneOf(['healthy', 'partial']);
});
```

---

These examples provide comprehensive coverage for integrating with the Pneumonia Detection API across multiple programming languages and platforms. Choose the implementation that best fits your technology stack and requirements.
