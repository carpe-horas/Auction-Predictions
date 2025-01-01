document.addEventListener("DOMContentLoaded", () => {
    console.log("qna 페이지 로드 완료");
});

function toggleAnswer(element) {
    const answer = element.nextElementSibling; 
    const icon = element.querySelector(".toggle-icon"); 

    // 다른 답변 닫음
    const allAnswers = document.querySelectorAll('.faq-answer');
    allAnswers.forEach(item => {
        if (item !== answer && item.classList.contains('active')) {
            item.classList.remove('active');
            item.style.maxHeight = null; 
            item.previousElementSibling.querySelector(".toggle-icon").textContent = "+"; 
        }
    });

    // 현재 클릭한 답변 토글
    if (answer.classList.contains("active")) {
        // 닫기 애니메이션
        answer.style.maxHeight = null; 
        answer.classList.remove("active");
        icon.textContent = "+"; 
    } else {
        // 열기 애니메이션
        answer.style.maxHeight = answer.scrollHeight + "px"; // 내용 높이만큼 설정
        answer.classList.add("active");
        icon.textContent = "-"; 
    }
}
