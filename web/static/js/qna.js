document.addEventListener("DOMContentLoaded", () => {
    console.log("qna 페이지 로드 완료");
});


function toggleAnswer(element) {
    const answer = element.nextElementSibling; // 답변 영역
    const icon = element.querySelector(".toggle-icon"); // 아이콘

    if (answer.classList.contains("active")) {
        // 닫기 애니메이션
        answer.style.maxHeight = null; // 높이 초기화
        answer.classList.remove("active");
        icon.textContent = "+"; 
    } else {
        // 열기 애니메이션
        answer.style.maxHeight = answer.scrollHeight + "px"; // 내용 높이만큼 설정
        answer.classList.add("active");
        icon.textContent = "-"; 
    }
}
