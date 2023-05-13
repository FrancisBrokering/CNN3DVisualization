const grid = document.getElementById("grid");
const submitButton = document.getElementById("submit");
let isDrawing = false;

function createGrid() {
    for (let i = 0; i < 28 * 28; i++) {
        const pixel = document.createElement("div");
        pixel.classList.add("pixel");
        pixel.addEventListener("mousedown", () => {
            isDrawing = true;
            pixel.classList.add("active");
        });
        pixel.addEventListener("mouseenter", () => {
            if (isDrawing) {
                pixel.classList.add("active");
            }
        });
        grid.appendChild(pixel);
    }
}

function getMatrix() {
    const matrix = [];
    for (let i = 0; i < 28; i++) {
        const row = [];
        for (let j = 0; j < 28; j++) {
            const pixel = grid.children[i * 28 + j];
            row.push(pixel.classList.contains("active") ? 1 : 0);
        }
        matrix.push(row);
    }
    return matrix;
}

submitButton.addEventListener("click", () => {
    const matrix = getMatrix();
    console.log(matrix);
    // process the matrix and send it the the CNN
});

// Stop drawing when the mouse is released
document.addEventListener("mouseup", () => {
    isDrawing = false;
});

createGrid();
