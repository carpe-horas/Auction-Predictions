document.addEventListener("DOMContentLoaded", () => {
    const itemSelect = document.getElementById('item-select'); // 셀렉트 박스
    const chartTitle = document.getElementById('chart-title'); // 분석 결과 제목
    const menuItems = document.querySelectorAll('.menu-item'); // 분석 메뉴 항목
    const chartCanvas = document.getElementById("chart");
    let chartInstance;

    let selectedItem = itemSelect.value; // 초기 품목 값
    let selectedMenu = ''; // 선택된 분석 메뉴


     // 품목이 선택될 때마다 제목 갱신
     itemSelect.addEventListener('change', () => {
        selectedItem = itemSelect.value; 
        updateChartTitle(); 
    });

    // 메뉴 항목 클릭 시 제목 갱신
    menuItems.forEach(menuItem => {
        menuItem.addEventListener('click', () => {
            selectedMenu = menuItem.textContent; 
            updateChartTitle(); 
        });
    });

    // 제목 갱신
    function updateChartTitle() {
        if (selectedItem && selectedMenu) {
            chartTitle.textContent = `${selectedItem} ${selectedMenu} 분석 결과`;
        } else {
            chartTitle.textContent = '분석 결과'; 
        }
    }

    // 페이지 로드 시 제목 갱신
    updateChartTitle();
    
    menuItems.forEach(item => {
        item.addEventListener("click", () => {
            const itemName = document.getElementById("item-select").value;
            const analysisType = item.getAttribute("data-type");

            if (!itemName) {
                alert("먼저 품목을 선택해주세요.");
                return;
            }

            fetch(`/analysis/get-item-data-v2?item=${itemName}&analysis_type=${analysisType}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                        return;
                    }

                    if (chartInstance) {
                        chartInstance.destroy();
                    }

                    let labels = [];
                    let values = [];
                    let secondaryValues = [];

                    if (analysisType === "daily_quantity_price_trend") {
                        // 일별 물량 및 가격 추세
                        labels = data.analysis_result.date;
                        values = data.analysis_result.average_price;
                        secondaryValues = data.analysis_result.total_quantity;

                        chartInstance = new Chart(chartCanvas, {
                            type: "bar", // 기본 타입은 bar
                            data: {
                                labels: labels,
                                datasets: [
                                    {
                                        label: "평균 단가",
                                        data: values, // 단가 데이터
                                        type: "line", 
                                        borderColor: "rgb(241, 90, 105)", 
                                        backgroundColor: "rgba(90, 120, 241, 0.2)", 
                                        yAxisID: "y1", // 단가는 오른쪽 Y축
                                        tension: 0.4, // 꺾은선 부드럽게
                                        pointRadius: 3, // 데이터 포인트 크기
                                        pointHoverRadius: 5 // 호버 시 데이터 포인트 크기
                                    },
                                    {
                                        label: "총 물량",
                                        data: secondaryValues,
                                        backgroundColor: "rgba(136, 8, 136, 0.6)", 
                                        yAxisID: "y" // 물량은 왼쪽 Y축
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                plugins: {
                                    legend: { display: true },
                                    tooltip: { enabled: true } 
                                },
                                scales: {
                                    x: {
                                        title: {
                                            display: false,
                                            text: "날짜 또는 구간"
                                        }
                                    },
                                    y: {
                                        beginAtZero: true,
                                        title: {
                                            display: true,
                                            text: "총 물량"
                                        },
                                        position: "left" 
                                    },
                                    y1: {
                                        beginAtZero: true,
                                        title: {
                                            display: true,
                                            text: "평균 단가"
                                        },
                                        position: "right", 
                                        grid: {
                                            drawOnChartArea: false // 오른쪽 Y축 격자선 제거
                                        },
                                        ticks: {
                                            callback: function(value) {
                                                return value.toLocaleString(); // 숫자 포맷팅
                                            }
                                        },
                                        suggestedMax: Math.max(...values) * 2 // 단가 데이터 범위 확장
                                    }
                                }
                            }
                        });
                        
                        return; // "일별 물량 및 가격 추세" 처리 종료
                    }

                    // 다른 분석 유형 처리
                    if (analysisType === "quantity_price_correlation") {
                        const scatterData = data.analysis_result.scatter_data;
                        if (!scatterData || scatterData.length === 0) {
                            alert("상관관계 데이터를 가져올 수 없습니다.");
                            return;
                        }
                        const scatterPoints = scatterData.map(point => ({
                            x: point.total_quantity,
                            y: point.average_price
                        }));

                        chartInstance = new Chart(chartCanvas, {
                            type: "scatter",
                            data: {
                                datasets: [{
                                    label: "물량 대비 평균 단가",
                                    data: scatterPoints,
                                    backgroundColor: "rgba(180, 76, 250, 0.8)",
                                    borderColor: "rgb(89, 198, 241)",
                                    borderWidth: 1,
                                    pointRadius: 5,
                                    pointHoverRadius: 8
                                }]
                            },
                            options: {
                                responsive: true,
                                plugins: {
                                    legend: { display: false },
                                    tooltip: { enabled: true }
                                },
                                scales: {
                                    x: {
                                        beginAtZero: true,
                                        title: {
                                            display: true,
                                            text: "총 물량"
                                        }
                                    },
                                    y: {
                                        beginAtZero: true,
                                        title: {
                                            display: true,
                                            text: "평균 단가"
                                        }
                                    }
                                }
                            }
                        });
                        return;
                    }

                    if (analysisType === "item_price_extremes") {
                        labels = ["최고가", "최저가"];
                        values = [data.analysis_result.max_price, data.analysis_result.min_price];
                    } else if (analysisType === "seasonal_average_price_quantity") {
                        labels = data.analysis_result.season;
                        values = data.analysis_result.average_price;
                        secondaryValues = data.analysis_result.total_quantity;
                    } else if (analysisType === "monthly_average_price_quantity") {
                        labels = data.analysis_result.month.map(m => `${m}월`);
                        values = data.analysis_result.average_price;
                        secondaryValues = data.analysis_result.total_quantity;
                    } else if (analysisType === "recent_30_days_average_price_quantity") {
                        labels = data.analysis_result.date;
                        values = data.analysis_result.average_price;
                        secondaryValues = data.analysis_result.total_quantity;
                    } else if (analysisType === "temperature_average_price_quantity") {
                        labels = data.analysis_result.temperature_range;
                        values = data.analysis_result.average_price;
                        secondaryValues = data.analysis_result.total_quantity;
                    } else if (analysisType === "rainfall_average_price_quantity") {
                        labels = data.analysis_result.rainfall_range;
                        values = data.analysis_result.average_price;
                        secondaryValues = data.analysis_result.total_quantity;
                    }

                    chartInstance = new Chart(chartCanvas, {
                        type: "bar",
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: "평균 단가",
                                    data: values,
                                    backgroundColor: "rgb(188, 255, 30)",
                                    yAxisID: "y1"
                                },
                                ...(secondaryValues.length > 0 ? [{
                                    label: "총 물량",
                                    data: secondaryValues,
                                    backgroundColor: "rgb(255, 6, 89)",
                                    yAxisID: "y"
                                }] : [])
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: { display: true },
                                tooltip: { enabled: true }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: false,
                                        text: "날짜 또는 구간"
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: "총 물량"
                                    },
                                    position: "left"
                                },
                                y1: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: "평균 단가"
                                    },
                                    position: "right",
                                    grid: {
                                        drawOnChartArea: false
                                    }
                                }
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error("데이터 요청 오류:", error);
                    alert("데이터를 가져오는 중 문제가 발생했습니다.");
                });
        });
    });
});
