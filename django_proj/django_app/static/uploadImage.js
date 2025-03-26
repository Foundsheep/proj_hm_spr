document.addEventListener('DOMContentLoaded', function() {
    const fileUpload = document.getElementById('file-upload');
    const fileInput = document.getElementById('file-input');
    const fileSelectBtn = document.getElementById('file-select-btn');
    const fileDrag = document.getElementById('file-drag');
    const filePreview = document.getElementById('file-preview');
    const fileList = document.getElementById('file-list');
    const uploadBtn = document.getElementById('upload-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultContainer = document.getElementById('segmentation-results');

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

        // Add files to list
        Array.from(files).forEach(file => {
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
            listItem.innerHTML = `
                ${file.name}
                <span class="badge bg-primary rounded-pill">${(file.size / 1024).toFixed(1)} KB</span>
            `;
            fileList.appendChild(listItem);
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

        const url = "/seg/"
        const response = await fetch(url, {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": csrftoken,
            }
        })
        
        // if  (!response.ok) {
        //     console.log(`Reponse status: ${response.ok}`);
        //     alaert(`Upload failed with status; ${response.status}`);
        // }
        
        // const content = await response.json();
        // resultContainer.classList.remove("d-none")
        // resultContainer.classList.add("d-flex")
        // resultContainer.innerHTML = `<div>
        // <img src="data:image/png;base64,${content["result"]}" width=300 height=200>
        // </div>`
        // alert(`${files.length} file(s) uploaded successfully`);

        // // Show loading state
        // resultContainer.innerHTML = "<div class='spinner-border text-primary' role='status'><span class='visually-hidden'>Loading...</span></div>";

        // fetch('/process-segmentation/', {
        //     method: 'POST',
        //     body: formData
        // })
        // .then(response => response.json())
        // .then(result => {
        //     // Clear previous results
        //     resultContainer.innerHTML = "";
        // })
        // .catch(error => {
        //     console.error('Error:', error);
        // });
    }

    function resetFiles() {
        location.reload();
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