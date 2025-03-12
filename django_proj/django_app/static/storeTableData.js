let currentIndex = 0;
const tables = document.querySelectorAll(".table-wrapper");

function showTable(index) {
  tables.forEach((table, i) => {
    table.style.display = i === index ? "block" : "none";
  });
}

function prevTable() {
  if (currentIndex > 0) {
    currentIndex--;
    showTable(currentIndex);
  }
}

function nextTable() {
  if (currentIndex < tables.length - 1) {
    currentIndex++;
    showTable(currentIndex);
  }
}

function storeSelectedTables() {
  const cumulativeTableBody = document.getElementById("cumulativeTableBody");
  const selectedTables = document.querySelectorAll(".table-wrapper");

  selectedTables.forEach((tableWrapper) => {
    const checkbox = tableWrapper.querySelector(".table-checkbox");
    if (checkbox.checked) {
      const clonedRows = tableWrapper.querySelector("table tbody").cloneNode(true);
      cumulativeTableBody.appendChild(clonedRows);
      checkbox.checked = false; // Uncheck after storing
    }
  });
}

function refreshPage() {
  location.reload();
}

// Initialize first table visibility
showTable(currentIndex);