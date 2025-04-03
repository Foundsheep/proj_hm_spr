document.addEventListener('DOMContentLoaded', function() {
    const fileUpload = document.getElementById('file-upload');
    const fileInput = document.getElementById('file-input');
    const fileSelectBtn = document.getElementById('file-select-btn');
    const fileDrag = document.getElementById('file-drag');
    const filePreview = document.getElementById('file-preview');
    const fileList = document.getElementById('file-list');
    const uploadBtn = document.getElementById('upload-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultContainer = document.getElementById('result-container');
    const segmentedImage = document.getElementById("segmented-image");
    const originalImage = document.getElementById("original-image");
    const fileName = document.getElementById("file-name");
    const pageIndicator = document.getElementById("page-indicator");
    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");
    const downloadBtn = document.getElementById("download-btn");

    let segmentedImages = [];
    let originalImageURLs = [];
    let fileNames = [];
    let currentIndex = 0;
    let imageSelections = [];

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileUpload.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        fileUpload.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileUpload.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    fileUpload.addEventListener('drop', handleDrop, false);

    // Handle file selection button
    fileSelectBtn.addEventListener('click', () => fileInput.click());

    // Handle file input change
    fileInput.addEventListener('change', handleFiles, false);

    // Upload button
    uploadBtn.addEventListener('click', uploadFiles);

    // Reset button
    resetBtn.addEventListener('click', resetFiles);

    prevBtn.addEventListener("click", () => {
        if (currentIndex > 0) {
            currentIndex--;
            showByIndex(currentIndex);
        }
    });

    nextBtn.addEventListener("click", () => {
        if (currentIndex < segmentedImages.length - 1) {
            currentIndex++;
            showByIndex(currentIndex);
        }
    });

    downloadBtn.addEventListener("click", async () => {
        if (segmentedImages.length === 0) {
            alert("No images to download");
        }

        const zip = new JSZip();
        const folder = zip.folder("segmented_images");

        for (let i = 0; i < segmentedImages.length; i++) {

            // skip if not selected
            if (!imageSelections[i]) continue;

            const base64 = segmentedImages[i];
            const byteString = atob(base64);
            const arrayBuffer = new Uint8Array(byteString.length);

            for (let j = 0; j < byteString.length; j++) {
                arrayBuffer[j] = byteString.charCodeAt(j);
            }

            folder.file(fileNames[i], arrayBuffer);
        }

        const content = await zip.generateAsync({ type: "blob" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(content);
        link.download = "segmented_images.zip";
        link.click();
    });

    // when event is triggered, this lets the event to execute its default action and stops it to be propagated further
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        fileUpload.classList.add('dragover');
    }

    function unhighlight() {
        fileUpload.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        // If called from input change, get files from event
        if (files instanceof Event) {
            files = fileInput.files;
        }

        // Clear previous list
        fileList.innerHTML = '';

        // Show preview area
        filePreview.classList.remove('d-none');
        fileDrag.classList.add('d-none');


        // Clear previous arrays
        originalImageURLs = [];
        fileNames = [];

        // Add files to list
        Array.from(files).forEach((file, index) => {
            // File name to display later
            fileNames.push(file.name);

            // Images are selected by default
            imageSelections.push(true);

            // UI list
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
            listItem.innerHTML = `
                ${file.name}
                <span class="badge bg-primary rounded-pill">${(file.size / 1024).toFixed(1)} KB</span>
            `;
            fileList.appendChild(listItem);

            // Original image url to display later
            const reader = new FileReader();
            reader.onload = function (e) {
                originalImageURLs[index] = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    async function uploadFiles() {
        const files = fileInput.files;
        if (files.length === 0) {
            alert('Please select files to upload');
            return;
        } else if (files.length > 10) {
            alert("images cannot be uploaded with more than 10 iamges");
            return;
        }

        const csrftoken = getCookie('csrftoken');
        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('images', file, file.name);
        });

        currentIndex = 0;
        const response = await fetch("/api/segment/", {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": csrftoken,
            }
        })
        
        if  (!response.ok) {
            console.log(`Reponse status: ${response.ok}`);
            alaert(`Upload failed with status; ${response.status}`);
        }
        
        const content = await response.json();
        if (content.result !== "success") {
            alert("Segmentation failed. Please try again.");
            return;
        }
        segmentedImages = content.images;
        resultContainer.classList.remove("d-none");
        resultContainer.classList.add("d-flex");
        
        showByIndex(currentIndex);
    }

    function resetFiles() {
        location.reload();
    }

    function showImage(index) {
        if (segmentedImages.length === 0) return;
        segmentedImage.src = `data:image/png;base64,${segmentedImages[index]}`;
        pageIndicator.textContent = `${index + 1} / ${segmentedImages.length}`;
        prevBtn.disabled = index === 0;
        nextBtn.disabled = index === segmentedImages.length - 1;
    }

    function showOriginalImage(index) {
        originalImage.src = originalImageURLs[index];
    }
    function showFileName(index) {
        fileName.textContent = fileNames[index];
    }

    function showByIndex(index) {
        showOriginalImage(index);
        showImage(index);
        showFileName(index);

        // checkbox checked = true/false updated
        const checkbox = document.getElementById("image-select");
        checkbox.checked = imageSelections[index];

        checkbox.onchange = () => {
            imageSelections[index] = checkbox.checked;
        };
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();

                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});