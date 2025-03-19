function validateMiddlePlate() {
    const middleName = document.getElementById("middle_name").value;
    const middleThickness = document.getElementById("middle_thickness").value;

    if ((middleName === "no-value" && middleThickness !== "no-value") ||
        (middleName !== "no-value" && middleThickness === "no-value")) {
        alert("중판의 이름과 두께는 둘 다 '선택 안 함' 혹은 선택되어야 합니다.");
        return false;  // Prevent form submission
    }
    return true;  // Allow form submission
}