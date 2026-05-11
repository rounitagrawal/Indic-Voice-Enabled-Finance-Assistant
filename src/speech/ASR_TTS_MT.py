import requests
import json

def ASR_config_call(language):

    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

    payload = json.dumps({
    "pipelineTasks": [
        {
        "taskType": "asr",
        "config": {
            "language": {
            "sourceLanguage": language
            }
        }
        }
    ],
    "pipelineRequestConfig": {
        "pipelineId": "64392f96daac500b55c543cd"
    }
    })
    headers = {
    'ulcaApikey': '023043f5a8-232b-4d97-8211-3f02e847b03c',
    'userID': 'fab68bae1dc648158911546d695959c4',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    text = response.text
    json_object = json.loads(text)
    serviceID = json_object['pipelineResponseConfig'][0]['config'][0]['serviceId']
    inference_api_key = json_object['pipelineInferenceAPIEndPoint']['inferenceApiKey']['value']

    return serviceID,inference_api_key


def ASR_call(language,audio_64):

    url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    serviceID , infer_key = ASR_config_call(language)

    payload = json.dumps({
    "pipelineTasks": [
        {
        "taskType": "asr",
        "config": {
            "language": {
            "sourceLanguage": language
            },
            "serviceId": serviceID,
            "audioFormat": "wav",
            "samplingRate": 16000
        }
        }
    ],
    "inputData": {
        "audio": [
        {
            "audioContent": audio_64
        }
        ]
    }
    })
    headers = {
    'Authorization': 'vAhBOFg8AT_gDkcevrkxRtwTygQIKIIYaYBhhuBcg9gmJ530FXYI35dNUg3r7miD', #infer_key
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.text
    # print("response",res)
    json_obj = json.loads(res)
    # print("json obj",json_obj)
    out_text = json_obj['pipelineResponse'][0]['output'][0]['source']
    return out_text


def MT_config_call(source_language,target_language):

    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

    payload = json.dumps({
    "pipelineTasks": [
        {
        "taskType": "translation",
        "config": {
            "language": {
            "sourceLanguage": source_language,
            "targetLanguage": target_language
            }
        }
        }
    ],
    "pipelineRequestConfig": {
        "pipelineId": "64392f96daac500b55c543cd"
    }
    })
    headers = {
    'ulcaApikey': '023043f5a8-232b-4d97-8211-3f02e847b03c',
    'userID': 'fab68bae1dc648158911546d695959c4',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    text = response.text
    json_object = json.loads(text)
    # serviceID = json_object['pipelineResponseConfig'][0]['config'][0]['serviceId']
    inference_api_key = json_object['pipelineInferenceAPIEndPoint']['inferenceApiKey']['value']


    return inference_api_key


def MT_call(source_lan,target_lan,input_text):

    url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

    serviceID = "ai4bharat/indictrans-v2-all-gpu--t4"
    infer_key = MT_config_call(source_lan,target_lan)

    payload = json.dumps({
    "pipelineTasks": [
        {
        "taskType": "translation",
        "config": {
            "language": {
            "sourceLanguage": source_lan,
            "targetLanguage": target_lan
            },
            "serviceId": serviceID
        }
        }
    ],
    "inputData": {
        "input": [
        {
            "source": input_text
        }
        ],
        "audio": [
        {
            "audioContent": None
        }
        ]
    }
    })
    headers = {
    'Authorization': 'vAhBOFg8AT_gDkcevrkxRtwTygQIKIIYaYBhhuBcg9gmJ530FXYI35dNUg3r7miD', #infer_key
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.text
    json_obj = json.loads(res)
    out_text = json_obj['pipelineResponse'][0]['output'][0]['target']

    return out_text


def TTS_config_call(target_lan):

    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

    payload = json.dumps({
    "pipelineTasks": [
        {
        "taskType": "tts",
        "config": {
            "language": {
            "sourceLanguage": target_lan
            }
        }
        }
    ],
    "pipelineRequestConfig": {
        "pipelineId": "64392f96daac500b55c543cd"
    }
    })
    headers = {
    'userID': 'fab68bae1dc648158911546d695959c4',
    'ulcaApikey': '023043f5a8-232b-4d97-8211-3f02e847b03c',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    text = response.text
    json_object = json.loads(text)
    serviceID = json_object['pipelineResponseConfig'][0]['config'][0]['serviceId']
    infer_key = json_object['pipelineInferenceAPIEndPoint']['inferenceApiKey']['value']

    return serviceID, infer_key

def TTS_call(language,input_text):

    url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

    serviceID,infer_key = TTS_config_call(language)
    payload = json.dumps({
    "pipelineTasks": [
        {
        "taskType": "tts",
        "config": {
            "language": {
            "sourceLanguage": language
            },
            "serviceId": serviceID,
            "gender": "female"
        }
        }
    ],
    "inputData": {
        "input": [
        {
            "source": input_text
        }
        ],
        "audio": [
        {
            "audioContent": None
        }
        ]
    }
    })
    headers = {
    'Authorization': 'vAhBOFg8AT_gDkcevrkxRtwTygQIKIIYaYBhhuBcg9gmJ530FXYI35dNUg3r7miD',#infer_key
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print("response",response.text)
    json_obj = response.json()
    # print(json_obj)
    out_audio = json_obj['pipelineResponse'][0]['audio'][0]['audioContent']
    return out_audio

