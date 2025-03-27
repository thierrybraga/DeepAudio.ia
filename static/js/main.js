document.addEventListener("DOMContentLoaded", function () {
  // Exibe o nome do arquivo selecionado (opcional)
  const fileInputs = document.querySelectorAll('input[type="file"]');
  fileInputs.forEach(input => {
    input.addEventListener("change", function () {
      const fileName = this.files[0]?.name;
      if (fileName) {
        let label = this.nextElementSibling;
        if (!label || !label.classList.contains("file-name")) {
          label = document.createElement("span");
          label.classList.add("file-name", "ms-2");
          this.parentNode.appendChild(label);
        }
        label.textContent = fileName;
      }
    });
  });

  // Exibe overlay de carregamento ao submeter formulários
  const forms = document.querySelectorAll("form");
  forms.forEach(form => {
    form.addEventListener("submit", function(){
      document.getElementById("loadingOverlay").style.display = "flex";
    });
  });
});
