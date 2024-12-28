document.addEventListener("DOMContentLoaded", () => {
    console.log("analysis 페이지 로드 완료");
});

document.addEventListener("DOMContentLoaded", async () => {
    const itemName = "배추"; // 요청할 품목 이름
    try {
        // Flask API로 데이터 요청
        const response = await fetch(`/analysis/data?item=${encodeURIComponent(itemName)}`);
        if (!response.ok) throw new Error("분석 데이터를 가져오는 데 실패했습니다.");

        const data = await response.json();
        console.log("가져온 데이터:", data);

        if (data.length === 0) {
            console.warn("데이터가 비어 있습니다. 그래프를 생성할 수 없습니다.");
            return;
        }

        // 데이터 가공
        const labels = [...new Set(data.map(item => item.date))]; // 날짜
        console.log("라벨(날짜):", labels);

        const datasets = [{
            label: itemName,
            data: labels.map(date => {
                const record = data.find(d => d.date === date);
                return record ? record.avg_unit_price_per_kg : null;
            }),
            borderColor: getRandomColor(),
            borderWidth: 2,
            fill: false
        }];
        console.log("생성된 데이터셋:", datasets);

        // Chart.js 그래프 생성
        const ctx = document.getElementById("priceChart").getContext("2d");
        new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: "top"
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "날짜"
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: "평균 단가 (원/kg)"
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error("분석 데이터를 불러오는 중 오류가 발생했습니다:", error);
    }
});

function getRandomColor() {
    return `hsl(${Math.random() * 360}, 70%, 70%)`;
}