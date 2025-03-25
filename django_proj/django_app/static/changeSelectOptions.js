document.addEventListener("DOMContentLoaded", function () {
    // Load plate names from Django
    const plateNameDict = JSON.parse(document.getElementById("plate_name_dict").textContent.trim());

    function updateSelectOptions(type) {
        const isAluminum = document.getElementById(`${type}_aluminum`).checked;
        const isSteel = document.getElementById(`${type}_steel`).checked;
        const isSteelCoated = document.getElementById(`${type}_is_coated`).checked;
        const nameSelectElement = document.getElementById(`${type}_name`);

        let options = [];

        if (isAluminum) {
            options = plateNameDict.aluminum;
        } else if (isSteel && isSteelCoated) {
            options = plateNameDict.steel_coated;
        } else {
            options = plateNameDict.steel_uncoated;
        }
        
        // Clear and update options
        if (type === "middle") {
            nameSelectElement.innerHTML = "<option value='no-value' selected>선택 안 함</option>";
        } else {
            nameSelectElement.innerHTML = "";
        }
        options.forEach(option => {
            const opt = document.createElement("option");
            opt.value = option;
            opt.textContent = option;
            nameSelectElement.appendChild(opt);
        });
    }

    function toggleCheckboxes(type) {
        const isAluminum = document.getElementById(`${type}_aluminum`).checked;
        const coatedCheckbox = document.getElementById(`${type}_is_coated`);
        const coatedLabel = document.querySelector(`label[for='${type}_is_coated']`);

        if (isAluminum) {
            coatedCheckbox.disabled = true;
            coatedCheckbox.checked = false;
            coatedLabel.classList.add("disabled-label");
        } else {
            coatedCheckbox.disabled = false;
            coatedLabel.classList.remove("disabled-label");
        }
    }

    function attachEventListeners(type) {
        document.getElementById(`${type}_aluminum`).addEventListener("change", () => {
            toggleCheckboxes(type);
            updateSelectOptions(type);
        });

        document.getElementById(`${type}_steel`).addEventListener("change", () => {
            toggleCheckboxes(type);
            updateSelectOptions(type);
        });

        document.getElementById(`${type}_is_coated`).addEventListener("change", () => {
            updateSelectOptions(type);
        });

        // Initial setup on page load
        toggleCheckboxes(type);
        updateSelectOptions(type);
    }

    // function validateMiddlePlate(event) {
    //     const middlePlateName = document.getElementById("middle_name").value;
    //     const middlePlateThickness = document.getElementById("middle_thickness").value;

    //     const bothAligned = (middlePlateName === "no-value" && middlePlateThickness === "no-value") || 
    //                         (middlePlateName !== "no-value" && middlePlateThickness !== "no-value");

    //     console.log(middlePlateName);
    //     console.log(middlePlateThickness);
    //     console.log(bothAligned);

    //     if (!bothAligned) {
    //         alert("중판의 이름과 두께는 둘 다 '선택 안 함' 혹은 선택되어야 합니다.");
    //         event.preventDefault(); // Prevent form submission
    //     }
    // }

    // Attach event listeners for both top and middle elements
    attachEventListeners("top");
    attachEventListeners("middle");
    attachEventListeners("bottom");

    // Prevent form submission if middle plate values are misaligned
    // document.querySelector("form").addEventListener("submit", validateMiddlePlate);
});
