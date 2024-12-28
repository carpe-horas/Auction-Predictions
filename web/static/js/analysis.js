document.addEventListener('DOMContentLoaded', () => {
    const itemSelect = document.getElementById('item-select');
    const chartContainer = document.getElementById('chart-container');
    const generateChartButton = document.getElementById('generate-chart');
    const loadingMessage = document.getElementById('loading-message');
    const footer = document.querySelector('footer');
    const mainContent = document.querySelector('main');

    // Footer 위치 조정 함수
    const adjustFooterSpacing = () => {
        const mainHeight = mainContent.offsetHeight;
        const viewportHeight = window.innerHeight;
        const footerHeight = footer.offsetHeight;

        if (mainHeight + footerHeight < viewportHeight) {
            footer.style.marginTop = `${viewportHeight - mainHeight - footerHeight}px`;
        } else {
            footer.style.marginTop = '150px';
        }
    };

    adjustFooterSpacing();
    window.addEventListener('resize', adjustFooterSpacing);

    // 로딩 메시지 표시 함수
    const showLoading = () => {
        loadingMessage.classList.remove('hidden');
        loadingMessage.classList.add('visible');
    };

    const hideLoading = () => {
        loadingMessage.classList.remove('visible');
        loadingMessage.classList.add('hidden');
    };

    if (itemSelect.options.length > 0) {
        generateChartButton.disabled = false;
    }

    generateChartButton.addEventListener('click', async () => {
        try {
            showLoading();

            const selectedItem = itemSelect.value;

            if (!selectedItem) {
                alert('먼저 품목을 선택하세요.');
                hideLoading();
                return;
            }

            const response = await fetch(`/analysis/get-item-data?item=${selectedItem}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            hideLoading();

            if (data.error) {
                alert(data.error);
                return;
            }

            chartContainer.innerHTML = '';

            await createCharts(data);

            adjustFooterSpacing();
        } catch (error) {
            hideLoading();
            console.error('Error during chart generation:', error);
            alert('시각화를 생성하는 중 오류가 발생했습니다.');
        }
    });

    async function createCharts(data) {
        const createChart = async (container, labels, datasets, title) => {
            const canvasElement = document.createElement('canvas');
    
            // 부모 컨테이너 크기를 기준으로 canvas 크기를 설정
            canvasElement.width = chartContainer.clientWidth * 0.9; // 차트 컨테이너 너비의 90%
            canvasElement.height = canvasElement.width / 2; // 2:1 비율
    
            container.appendChild(canvasElement);
    
            new Chart(canvasElement.getContext('2d'), {
                type: 'bar',
                data: {
                    labels,
                    datasets,
                },
                options: {
                    responsive: false, 
                    maintainAspectRatio: false, // 캔버스 비율 유지 비활성화
                    scales: {
                        y: { beginAtZero: true },
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: title,
                        },
                    },
                },
            });
        };
    
        // 지역별 분석 차트
        if (data.regional_analysis?.region?.length) {
            const regionalDiv = document.createElement('div');
            regionalDiv.style.marginBottom = '50px';
            chartContainer.appendChild(regionalDiv);
    
            await createChart(
                regionalDiv,
                data.regional_analysis.region,
                [
                    {
                        label: '평균물량 (kg)',
                        data: data.regional_analysis.avg_quantity_kg,
                        backgroundColor: 'rgba(135, 206, 250, 0.7)',
                    },
                    {
                        label: '평균단가 (원/kg)',
                        data: data.regional_analysis.avg_unit_price_per_kg,
                        type: 'line',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2,
                        fill: false,
                    },
                ],
                '지역별 분석'
            );
        }
    
        // 계절별 분석 차트
        if (data.seasonal_analysis?.season?.length) {
            const seasonalDiv = document.createElement('div');
            seasonalDiv.style.marginBottom = '50px';
            chartContainer.appendChild(seasonalDiv);
    
            await createChart(
                seasonalDiv,
                data.seasonal_analysis.season,
                [
                    {
                        label: '평균물량 (kg)',
                        data: data.seasonal_analysis.avg_quantity_kg,
                        backgroundColor: 'rgba(135, 206, 250, 0.7)',
                    },
                    {
                        label: '평균단가 (원/kg)',
                        data: data.seasonal_analysis.avg_unit_price_per_kg,
                        type: 'line',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2,
                        fill: false,
                    },
                ],
                '계절별 분석'
            );
        }
    }
    
});
