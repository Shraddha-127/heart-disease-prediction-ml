// =====================================================================
// history.js — Search, sort, and pagination for the prediction history
// table. All operations run client-side over the rows rendered by the
// server (data is not refetched, so backend logic stays untouched).
// =====================================================================

document.addEventListener("DOMContentLoaded", function () {

    const table = document.getElementById("historyTable");
    if (!table) return;

    const tbody = table.querySelector("tbody");
    const searchInput = document.getElementById("historySearch");
    const sortSelect = document.getElementById("sortSelect");
    const paginationEl = document.getElementById("historyPagination");

    const ROWS_PER_PAGE = 8;
    let currentPage = 1;

    function getAllRows() {
        return Array.from(tbody.querySelectorAll("tr"));
    }

    function filterRows() {
        const query = (searchInput.value || "").toLowerCase();
        return getAllRows().filter((row) => {
            return row.textContent.toLowerCase().includes(query);
        });
    }

    function sortRows(rows) {
        const mode = sortSelect.value;
        const sorted = [...rows];

        sorted.sort((a, b) => {
            if (mode.startsWith("date")) {
                const dateA = new Date(a.getAttribute("data-date"));
                const dateB = new Date(b.getAttribute("data-date"));
                return mode === "date-desc" ? dateB - dateA : dateA - dateB;
            } else {
                const confA = parseFloat(a.getAttribute("data-confidence")) || 0;
                const confB = parseFloat(b.getAttribute("data-confidence")) || 0;
                return mode === "confidence-desc" ? confB - confA : confA - confB;
            }
        });

        return sorted;
    }

    function renderTable() {
        const filtered = sortRows(filterRows());
        const totalPages = Math.max(1, Math.ceil(filtered.length / ROWS_PER_PAGE));

        if (currentPage > totalPages) currentPage = totalPages;

        // Hide all rows first
        getAllRows().forEach((row) => (row.style.display = "none"));

        // Show only the rows for the current page
        const start = (currentPage - 1) * ROWS_PER_PAGE;
        const pageRows = filtered.slice(start, start + ROWS_PER_PAGE);
        pageRows.forEach((row) => (row.style.display = ""));

        // Re-append in sorted order so visually they appear correctly
        pageRows.forEach((row) => tbody.appendChild(row));

        renderPagination(totalPages);
    }

    function renderPagination(totalPages) {
        paginationEl.innerHTML = "";

        if (totalPages <= 1) return;

        for (let i = 1; i <= totalPages; i++) {
            const btn = document.createElement("button");
            btn.textContent = i;
            if (i === currentPage) btn.classList.add("active");
            btn.addEventListener("click", () => {
                currentPage = i;
                renderTable();
            });
            paginationEl.appendChild(btn);
        }
    }

    searchInput.addEventListener("input", () => {
        currentPage = 1;
        renderTable();
    });

    sortSelect.addEventListener("change", () => {
        currentPage = 1;
        renderTable();
    });

    renderTable();
});
