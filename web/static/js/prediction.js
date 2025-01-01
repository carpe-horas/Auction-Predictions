document.addEventListener("DOMContentLoaded", () => {
  const itemSelect = document.getElementById("item-select");
  const predictionDate = document.getElementById("prediction-date");
  const generateButton = document.getElementById("generate-prediction");
  const resultContainer = document.getElementById("result-container");
  const predictionSummary = document.getElementById("prediction-summary");
  const chartCanvas = document.getElementById("chart");
  const loadingMessage = document.getElementById("loading-message");
  let chartInstance;

  // Flatpickr 달력 커스터마이징
  flatpickr(predictionDate, {
      dateFormat: "Y-m-d", 
      defaultDate: "today", 
      minDate: "today", 
      locale: {
          firstDayOfWeek: 0,
      },
      theme: "light",
  });

  // 품목 데이터 가져오기
  async function fetchItems() {
      try {
          const response = await fetch("/prediction/items");
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          const items = data.items;

          if (!items || items.length === 0) {
              console.warn("품목 데이터가 비어 있습니다.");
              return;
          }

          items.forEach((item) => {
              const option = document.createElement("option");
              option.value = item;
              option.textContent = item;
              itemSelect.appendChild(option);
          });

          itemSelect.addEventListener("change", toggleGenerateButton);
      } catch (error) {
          console.error("품목 데이터를 가져오는 중 오류 발생:", error);
      }
  }

  // 버튼 활성화/비활성화
  function toggleGenerateButton() {
      generateButton.disabled = !(itemSelect.value && predictionDate.value);
  }

  // 예측 생성 버튼 클릭 시
  generateButton.addEventListener("click", function() {
      if (!itemSelect.value) {
          alert("품목을 먼저 선택하세요.");
          return; 
      }

      // 로딩 메시지 표시
      loadingMessage.classList.remove("hidden");
      console.log('로딩 메시지 표시')

      // 예측 처리 코드
      fetchPrediction();
  });

  // 예측 데이터 가져오기
  async function fetchPrediction() {
      const item = itemSelect.value;
      const date = predictionDate.value;

      if (!item || !date) {
          alert("품목과 날짜를 모두 선택하세요.");
          return;
      }

      try {
          const response = await fetch("/prediction/predict", {
              method: "POST",
              headers: {
                  "Content-Type": "application/json",
              },
              body: JSON.stringify({ item, date }),
          });

          if (!response.ok) {
              const errorDetails = await response.json();
              throw new Error(`HTTP error! status: ${response.status}, message: ${errorDetails.error || "알 수 없는 오류"}`);
          }

          const data = await response.json();
          renderPrediction(data);
      } catch (error) {
          console.error("예측 데이터를 가져오는 중 오류 발생:", error);
          alert(`예측 요청 중 오류 발생: ${error.message}`);
      }
  }

  // 예측 결과 렌더링
  function renderPrediction(data) {
      const { item, date, predicted_price, historical_data } = data;

      predictionSummary.textContent = `예측 결과 선택한 ${date}에 대한 ${item}의 예측 가격은 ${predicted_price.toFixed(2)}원입니다.`;

      resultContainer.classList.remove("hidden");

      if (chartInstance) {
          chartInstance.destroy();
      }

      chartInstance = new Chart(chartCanvas, {
          type: "line",
          data: {
              labels: historical_data.dates,
              datasets: [
                  {
                      label: "평균 가격 (원)",
                      data: historical_data.prices,
                      borderColor: "rgba(75, 192, 192, 1)",
                      backgroundColor: "rgba(75, 192, 192, 0.2)",
                      borderWidth: 2,
                  },
                  {
                      label: "단가변동성",
                      data: historical_data.volatilities,
                      borderColor: "rgba(255, 99, 132, 1)",
                      backgroundColor: "rgba(255, 99, 132, 0.2)",
                      borderWidth: 2,
                      yAxisID: "y-axis-2",
                  },
              ],
          },
          options: {
              responsive: true,
              plugins: { legend: { display: true } },
              scales: {
                  x: {
                      title: { display: true, text: "날짜" },
                  },
                  y: {
                      title: { display: true, text: "가격 (원)" },
                      beginAtZero: true,
                  },
                  "y-axis-2": {
                      position: "right",
                      title: { display: true, text: "단가변동성" },
                      grid: { drawOnChartArea: false },
                  },
              },
          },
      });

      // 로딩 메시지 숨기기
      loadingMessage.classList.add("hidden");
  }

  fetchItems();
});
