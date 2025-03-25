import os
import time
import logging
import argparse
import json
import requests
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import jwt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_workflow")


load_dotenv()


@dataclass
class WorkflowConfig:
    openai_api_key: str

    dalle_model: str = "dall-e-3"
    dalle_quality: str = "standard"
    dalle_size: str = "1024x1024"
    dalle_style: str = "vivid"
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: int = 5

    upscale_factor: int = 2
    upscale_method: str = "esrgan"
    upscale_quality: int = 95

    adobe_client_id: Optional[str] = None
    adobe_client_secret: Optional[str] = None
    adobe_api_key: Optional[str] = None
    adobe_technical_account_id: Optional[str] = None
    adobe_org_id: Optional[str] = None
    adobe_private_key_path: Optional[str] = None
    enable_adobe_upload: bool = True

    output_dir: Path = Path("output")
    generated_dir: Path = Path("output/generated")
    upscaled_dir: Path = Path("output/upscaled")


class DALLEImageGenerator:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.api_key = config.openai_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://api.openai.com/v1/images/generations"

        self.config.generated_dir.mkdir(parents=True, exist_ok=True)

    def generate_image(self, prompt: str) -> Optional[str]:
        retries = 0
        while retries < self.config.max_retries:
            try:
                payload = {
                    "model": self.config.dalle_model,
                    "prompt": prompt,
                    "n": 1,
                    "size": self.config.dalle_size,
                    "quality": self.config.dalle_quality,
                    "style": self.config.dalle_style
                }

                logger.info(f"Generating image for prompt: {prompt}")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 429:
                    logger.warning(
                        "Rate limit exceeded, waiting before retry...")
                    time.sleep(self.config.retry_delay * (retries + 1))
                    retries += 1
                    continue

                response.raise_for_status()
                data = response.json()

                if "data" in data and len(data["data"]) > 0:
                    image_url = data["data"][0]["url"]
                    return self._download_image(image_url, prompt)
                else:
                    logger.error(f"No image data in response: {data}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {str(e)}")
                if retries < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (retries + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                retries += 1

        logger.error(
            f"Failed to generate image after {self.config.max_retries} attempts")
        return None

    def _download_image(self, image_url: str, prompt: str) -> Optional[Path]:
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
            safe_prompt = safe_prompt[:50]
            filename = f"{int(time.time())}_{safe_prompt}.png"
            file_path = self.config.generated_dir / filename

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Image saved to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download image: {str(e)}")
            return None

    def generate_batch(self, prompts: List[str]) -> Dict[str, Path]:
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_prompt = {
                executor.submit(self.generate_image, prompt): prompt
                for prompt in prompts
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_prompt),
                total=len(prompts),
                desc="Generating images"
            ):
                prompt = future_to_prompt[future]
                try:
                    file_path = future.result()
                    if file_path:
                        results[prompt] = file_path
                except Exception as e:
                    logger.error(
                        f"Error processing prompt '{prompt}': {str(e)}")

        return results


class ImageUpscaler:

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.config.upscaled_dir.mkdir(parents=True, exist_ok=True)

        if self.config.upscale_method == "esrgan":

            self._setup_esrgan()

    def _setup_esrgan(self):
        try:

            self.esrgan_model_path = "models/ESRGAN.pb"

            if not os.path.exists(self.esrgan_model_path):
                logger.warning(
                    f"ESRGAN model not found at {self.esrgan_model_path}. "
                    "Downloading model..."
                )
                os.makedirs(os.path.dirname(
                    self.esrgan_model_path), exist_ok=True)

                model_url = "https://github.com/xinntao/ESRGAN/releases/download/v0.1.0/RRDB_ESRGAN_x4.pth"
                self._download_model(model_url, self.esrgan_model_path)

            self.esrgan_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.esrgan_model.readModel(self.esrgan_model_path)
            self.esrgan_model.setModel("esrgan", self.config.upscale_factor)

            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.esrgan_model.setPreferableBackend(
                    cv2.dnn.DNN_BACKEND_CUDA)
                self.esrgan_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("CUDA is available and will be used for upscaling")
            else:
                logger.info("CUDA not available, using CPU for upscaling")

        except Exception as e:
            logger.error(f"Failed to set up ESRGAN: {str(e)}")
            logger.info("Falling back to Lanczos upscaling method")
            self.config.upscale_method = "lanczos"

    def _download_model(self, url: str, path: str):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="Downloading ESRGAN model"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Model downloaded to {path}")
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise

    def upscale_image(self, image_path: Path) -> Optional[Path]:
        try:
            logger.info(f"Upscaling image: {image_path}")

            upscaled_path = self.config.upscaled_dir / \
                f"upscaled_{image_path.name}"

            if self.config.upscale_method == "esrgan":

                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"Could not read image: {image_path}")

                upscaled = self.esrgan_model.upsample(img)
                cv2.imwrite(str(upscaled_path), upscaled,
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])

            elif self.config.upscale_method in ["bicubic", "lanczos"]:

                with Image.open(image_path) as img:
                    width, height = img.size
                    new_width = width * self.config.upscale_factor
                    new_height = height * self.config.upscale_factor

                    resample_method = (
                        Image.BICUBIC if self.config.upscale_method == "bicubic"
                        else Image.LANCZOS
                    )

                    upscaled = img.resize(
                        (new_width, new_height), resample_method)
                    upscaled.save(
                        upscaled_path,
                        format="PNG",
                        quality=self.config.upscale_quality
                    )
            else:
                logger.error(
                    f"Unknown upscale method: {self.config.upscale_method}")
                return None

            logger.info(f"Image upscaled to {upscaled_path}")
            return upscaled_path

        except Exception as e:
            logger.error(f"Failed to upscale image {image_path}: {str(e)}")
            return None

    def upscale_batch(self, image_paths: List[Path]) -> Dict[Path, Path]:

        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_path = {
                executor.submit(self.upscale_image, path): path
                for path in image_paths
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_path),
                total=len(image_paths),
                desc="Upscaling images"
            ):
                input_path = future_to_path[future]
                try:
                    upscaled_path = future.result()
                    if upscaled_path:
                        results[input_path] = upscaled_path
                except Exception as e:
                    logger.error(f"Error upscaling {input_path}: {str(e)}")

        return results


class AdobeStockUploader:

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.access_token = self._get_jwt_token()
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "x-api-key": self.config.adobe_api_key,
            "Content-Type": "application/json"
        }
        self.upload_url = "https://stock-contributor.adobe.io/api/v1/files"

    def _get_jwt_token(self) -> str:
        try:
            with open(self.config.adobe_private_key_path, 'r') as key_file:
                private_key = key_file.read()

            expiry_time = int(time.time()) + 86400
            claims = {
                "exp": expiry_time,
                "iss": self.config.adobe_org_id,
                "sub": self.config.adobe_technical_account_id,
                "aud": f"https://ims-na1.adobelogin.com/c/{self.config.adobe_client_id}"
            }

            token = jwt.encode(claims, private_key, algorithm='RS256')

            exchange_url = "https://ims-na1.adobelogin.com/ims/exchange/jwt"
            payload = {
                "client_id": self.config.adobe_client_id,
                "client_secret": self.config.adobe_client_secret,
                "jwt_token": token
            }

            response = requests.post(exchange_url, data=payload)
            response.raise_for_status()

            access_token = response.json().get("access_token")
            if not access_token:
                raise ValueError("No access token in response")

            logger.info("Successfully obtained Adobe Stock access token")
            return access_token

        except Exception as e:
            logger.error(
                f"Failed to obtain Adobe Stock access token: {str(e)}")
            raise

    def _prepare_metadata(self, prompt: str) -> Dict[str, Any]:

        title = prompt[:50] if len(prompt) > 50 else prompt

        description = prompt

        keywords = [
            word.strip().lower() for word in prompt.replace(',', ' ').split()
            if len(word.strip()) > 2
        ]

        keywords = list(dict.fromkeys(keywords))[:50]

        return {
            "title": title,
            "description": description,
            "keywords": keywords,
            "categories": ["101"],
            "is_illustration": True,
            "is_editorial": False,
            "is_age_specific": False,
            "is_release_required": False,
            "content_type": "illustration/vector"
        }

    def upload_image(self, image_path: Path, prompt: str) -> Optional[str]:

        try:
            logger.info(f"Uploading image to Adobe Stock: {image_path}")

            metadata = self._prepare_metadata(prompt)
            init_payload = {
                "file_name": image_path.name,
                "content_type": "image/png",
                "metadata": metadata
            }

            response = requests.post(
                self.upload_url,
                headers=self.headers,
                json=init_payload
            )
            response.raise_for_status()

            upload_data = response.json()
            upload_url = upload_data.get("upload_url")
            asset_id = upload_data.get("asset_id")

            if not upload_url or not asset_id:
                raise ValueError(f"Invalid response: {upload_data}")

            with open(image_path, "rb") as image_file:
                file_upload_response = requests.put(
                    upload_url,
                    data=image_file,
                    headers={"Content-Type": "image/png"}
                )
                file_upload_response.raise_for_status()

            confirm_url = f"{self.upload_url}/{asset_id}/upload_finish"
            confirm_response = requests.post(confirm_url, headers=self.headers)
            confirm_response.raise_for_status()

            logger.info(
                f"Successfully uploaded image to Adobe Stock with ID: {asset_id}")
            return asset_id

        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {str(e)}")
            return None

    def upload_batch(
        self, images: Dict[Path, str]
    ) -> Dict[Path, Optional[str]]:

        results = {}
        for image_path, prompt in tqdm(
            images.items(), desc="Uploading to Adobe Stock"
        ):
            asset_id = self.upload_image(image_path, prompt)
            results[image_path] = asset_id

        return results


def read_prompts_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    """Main function to run the full workflow"""
    parser = argparse.ArgumentParser(
        description="Image generation, upscaling, and Adobe Stock upload workflow"
    )
    parser.add_argument(
        "--prompts-file",
        required=True,
        help="Path to a text file containing one prompt per line"
    )
    parser.add_argument(
        "--config",
        default=".env",
        help="Path to .env configuration file"
    )
    parser.add_argument(
        "--upscale-method",
        choices=["bicubic", "lanczos", "esrgan"],
        default="lanczos",
        help="Method to use for upscaling"
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=2,
        help="Factor by which to upscale images"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of images to process in parallel"
    )
    parser.add_argument(
        "--no-adobe-upload",
        action="store_true",
        help="Skip uploading to Adobe Stock"
    )

    args = parser.parse_args()

    config = WorkflowConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        adobe_client_id=os.getenv("ADOBE_CLIENT_ID"),
        adobe_client_secret=os.getenv("ADOBE_CLIENT_SECRET"),
        adobe_api_key=os.getenv("ADOBE_API_KEY"),
        adobe_technical_account_id=os.getenv("ADOBE_TECHNICAL_ACCOUNT_ID"),
        adobe_org_id=os.getenv("ADOBE_ORG_ID"),
        adobe_private_key_path=os.getenv("ADOBE_PRIVATE_KEY_PATH"),
        upscale_method=args.upscale_method,
        upscale_factor=args.upscale_factor,
        batch_size=args.batch_size,
        enable_adobe_upload=not args.no_adobe_upload
    )

    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY is required")
        return

    if config.enable_adobe_upload:
        if (not config.adobe_client_id or not config.adobe_client_secret or
                not config.adobe_api_key or not config.adobe_technical_account_id or
                not config.adobe_org_id or not config.adobe_private_key_path):
            logger.error("Adobe Stock API credentials are required for uploading. "
                         "Use --no-adobe-upload to skip uploading.")
            return

    try:
        prompts = read_prompts_from_file(args.prompts_file)
        if not prompts:
            logger.error("No prompts found in the file")
            return
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    except Exception as e:
        logger.error(f"Failed to read prompts file: {str(e)}")
        return

    image_generator = DALLEImageGenerator(config)
    upscaler = ImageUpscaler(config)

    generated_images = image_generator.generate_batch(prompts)
    logger.info(f"Generated {len(generated_images)} images")

    if not generated_images:
        logger.error("No images were generated")
        return

    upscaled_images = upscaler.upscale_batch(list(generated_images.values()))
    logger.info(f"Upscaled {len(upscaled_images)} images")

    if not upscaled_images:
        logger.error("No images were upscaled")
        return

    successful_uploads = 0
    uploaded_assets = {}

    if config.enable_adobe_upload:

        uploader = AdobeStockUploader(config)

        upload_mapping = {}
        for original_path, upscaled_path in upscaled_images.items():

            for prompt, gen_path in generated_images.items():
                if gen_path == original_path:
                    upload_mapping[upscaled_path] = prompt
                    break

        uploaded_assets = uploader.upload_batch(upload_mapping)
        successful_uploads = sum(
            1 for asset_id in uploaded_assets.values() if asset_id)
        logger.info(
            f"Successfully uploaded {successful_uploads} images to Adobe Stock")
    else:
        logger.info("Adobe Stock upload skipped (--no-adobe-upload flag used)")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_images": len(generated_images),
        "upscaled_images": len(upscaled_images),
        "adobe_upload_enabled": config.enable_adobe_upload,
        "uploaded_images": successful_uploads
    }

    if config.enable_adobe_upload and uploaded_assets:

        upload_mapping = {}
        for original_path, upscaled_path in upscaled_images.items():
            for prompt, gen_path in generated_images.items():
                if gen_path == original_path:
                    upload_mapping[upscaled_path] = prompt
                    break

        results["assets"] = [
            {
                "prompt": upload_mapping.get(path),
                "path": str(path),
                "asset_id": asset_id
            }
            for path, asset_id in uploaded_assets.items() if asset_id
        ]

    with open("workflow_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Workflow completed. Results saved to workflow_results.json")


if __name__ == "__main__":
    main()
