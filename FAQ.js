// FAQ Toggle Functionality
document.querySelectorAll(".faq-question").forEach((question) => {
  question.addEventListener("click", () => {
    const item = question.parentElement;
    const answer = item.querySelector(".faq-answer");

    // Close other open items
    document.querySelectorAll(".faq-item").forEach((otherItem) => {
      if (otherItem !== item) {
        otherItem.classList.remove("active");
        otherItem.querySelector(".faq-answer").classList.remove("active");
      }
    });

    // Toggle current item
    item.classList.toggle("active");
    answer.classList.toggle("active");
  });
});

// Search Functionality
const searchBox = document.getElementById("searchBox");
const faqItems = document.querySelectorAll(".faq-item");

searchBox.addEventListener("input", (e) => {
  const searchTerm = e.target.value.toLowerCase();

  faqItems.forEach((item) => {
    const question = item
      .querySelector(".faq-question")
      .textContent.toLowerCase();
    const answer = item.querySelector(".faq-answer").textContent.toLowerCase();

    if (question.includes(searchTerm) || answer.includes(searchTerm)) {
      item.style.display = "block";
    } else {
      item.style.display = "none";
    }
  });
});

// Category Filter Functionality
const tabs = document.querySelectorAll(".tab");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    // Update active tab
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");

    const category = tab.dataset.category;

    // Filter FAQ items
    faqItems.forEach((item) => {
      if (category === "all" || item.dataset.category === category) {
        item.style.display = "block";
      } else {
        item.style.display = "none";
      }
    });

    // Clear search when switching categories
    searchBox.value = "";
  });
});

// Fade in animation on load
window.addEventListener("load", () => {
  document.querySelectorAll(".fade-in").forEach((element, index) => {
    setTimeout(() => {
      element.style.animationDelay = `${index * 0.1}s`;
    }, 100);
  });
});
