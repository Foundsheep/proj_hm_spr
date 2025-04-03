document.addEventListener("DOMContentLoaded", function () {
    const resultContainer = document.getElementById("result-container");
    const outputBox = document.getElementById("output-box");
    const generatedImage = document.getElementById("generated-image");
    const previewSentence = document.getElementById("preview-sentence");
    const prevBtn = document.getElementById("prev-btn");
    const pageIndicator = document.getElementById("page-indicator");
    const nextBtn = document.getElementById("next-btn");
    const submitBtn = document.getElementById("submit-btn");
    const resetBtn = document.getElementById("reset-btn");
    const downloadBtn = document.getElementById("download-btn");

    let generatedImages = [];
    let currentIndex = 0;
    let formData;
    let imageSelections = [];

    // Submit button click
    submitBtn.addEventListener("click", async function () {
        const spinner = document.getElementById("loading-spinner");
        previewSentence.classList.add("d-none");
        spinner.classList.remove("d-none");
        
        // Call an API
        const form = submitBtn.closest("form");
        formData = new FormData(form);
        
        try {
            const response = await fetch("/api/generate/", {
                method: "POST",
                body: formData,
                headers: {
                    "X-CSRFToken": getCSRFToken(),
                },
            });
            
            const data = await response.json();
            spinner.classList.add("d-none");
            if (data.result !== "success") {                                
                alert("Generation failed. Please try again.");
                previewSentence.classList.remove("d-none");
                outputBox.classList.add("d-none");
                return;
            }

            // Swith on & off elements
            outputBox.classList.remove("d-none");
    
            if (data.images && Array.isArray(data.images)) {
                generatedImages = data.images;

                // Images are selected by default
                for (i = 0; i < generatedImages.length; i++) {
                    imageSelections.push(true);
                }

                // Refresh the index everytime called
                currentIndex = 0;
                showByIndex(currentIndex);
            } else {
                alert("이미지 생성 결과 없음");
            }
        } catch (error) {
            console.error("API error", error);
        }
    });


    function getCSRFToken() {
        return document.querySelector("[name=csrfmiddlewaretoken]").value;
    }

    function showByIndex(index) {
        if (generatedImages.length === 0) return;

        generatedImage.src = `data:image/png;base64,${generatedImages[index]}`;
        pageIndicator.textContent = `${index + 1} / ${generatedImages.length}`;
        prevBtn.disabled = index === 0;
        nextBtn.disabled = index === generatedImages.length - 1;

        // checkbox checked = true/false updated
        const checkbox = document.getElementById("image-select");
        checkbox.checked = imageSelections[index];

        checkbox.onchange = () => {
            imageSelections[index] = checkbox.checked;
        };
        
    }

    // Reset button
    resetBtn.addEventListener('click', () => {
        location.reload();
    });


    prevBtn.addEventListener("click", () => {
        if (currentIndex > 0) {
            currentIndex--;
            showByIndex(currentIndex);
        }
    });

    nextBtn.addEventListener("click", () => {
        if (currentIndex < generatedImages.length - 1) {
            currentIndex++;
            showByIndex(currentIndex);
        }
    });

    downloadBtn.addEventListener("click", async () => {
        if (generatedImages.length === 0) {
            alert("No images to download");
            return;
        }

        const zip = new JSZip();
        const folder = zip.folder("generated_images");

        for (let i = 0; i <generatedImages.length; i++) {

            // skip if not selected
            if (!imageSelections[i]) continue;

            // Set file name
            let fileName = [
                formData.get("rivet"),
                formData.get("die"),
                formData.get("upper_type"),
                formData.get("upper_thickness"),
                formData.get("middle_type"),
                formData.get("middle_thickness"),
                formData.get("lower_type"),
                formData.get("lower_thickness"),
                formData.get("head_height"),
                i + 1
            ];
            if (formData.get("middle_type") === null) {
                const upperPart = fileName.slice(0, 4);
                const lowerPart = fileName.slice(6);
                fileName = upperPart.concat(lowerPart);
            }
            fileName = fileName.join("_") + ".png";

            // Get images
            const base64 = generatedImages[i];
            const byteString = atob(base64);
            const arrayBuffer = new Uint8Array(byteString.length);

            for (let j = 0; j < byteString.length; j++) {
                arrayBuffer[j] = byteString.charCodeAt(j);
            }

            folder.file(fileName, arrayBuffer);
        }

        const content = await zip.generateAsync({ type: "blob" });

        const link = document.createElement("a");
        link.href = URL.createObjectURL(content);
        link.download = "generated_images.zip";
        link.click();
    });
})