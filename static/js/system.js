(function () {
    const dictionaries = {
        zh: {
            brandCaption: "锅炉灰污智能预测",
            navHome: "系统概览",
            navCleaning: "数据清洗",
            navPrediction: "消融实验",
            navOptimization: "优化过程",
            topTitle: "工业时序智能分析平台",
            logout: "退出登录",
            language: "EN",
            loginEyebrow: "PSBAF · INDUSTRIAL AI",
            loginHeroTitle: "让锅炉灰污趋势变得可预测",
            loginHeroDesc: "融合信号分解、深度时序模型与智能优化算法，为锅炉运行分析提供清晰、可信的预测依据。",
            loginTitle: "欢迎回来",
            adminLoginTitle: "管理员登录",
            loginSubtitle: "登录后进入锅炉灰污预测与分析平台。",
            username: "用户名",
            password: "密码",
            usernamePlaceholder: "请输入用户名",
            passwordPlaceholder: "请输入密码",
            loginButton: "登录系统",
            adminEntry: "管理员入口",
            userEntry: "普通用户入口",
            demoHint: "本地演示账号",
            heroKicker: "POWER PLANT INTELLIGENCE",
            heroTitle: "燃煤电站锅炉灰污预测系统",
            heroDesc: "集成小波去噪、VMD 多尺度分解、TSMixer / TiDE 时序预测与 WOA 智能优化，形成从数据处理到结果解释的一体化分析流程。",
            featureCleaning: "信号预处理",
            featureCleaningDesc: "通过软、硬阈值小波去噪改善清洁因子序列质量。",
            featurePrediction: "多模型预测",
            featurePredictionDesc: "对比 TSMixer、TiDE 及其 VMD 增强模型的预测性能。",
            featureOptimization: "智能参数寻优",
            featureOptimizationDesc: "利用 WOA 搜索关键超参数，并以三维视图呈现优化轨迹。",
            cleaningEyebrow: "SIGNAL PREPROCESSING",
            cleaningTitle: "数据清洗与质量评估",
            cleaningDesc: "对比小波软阈值与硬阈值去噪效果，结合残差能量比和平滑度指标评估信号质量。",
            softTitle: "软阈值小波去噪",
            softDesc: "保留信号连续性，获得更平滑的清洁因子变化趋势。",
            hardTitle: "硬阈值小波去噪",
            hardDesc: "直接滤除低于阈值的小波系数，突出主要信号结构。",
            rer: "残差能量比",
            smooth: "平滑度指数",
            predictionEyebrow: "ABLATION STUDY",
            predictionTitle: "预测模型消融实验",
            predictionDesc: "通过统一误差指标比较 VMD 分解和 WOA 优化对 TSMixer、TiDE 模型的贡献。",
            tsmixerTitle: "TSMixer 模型组",
            tsmixerDesc: "比较基础模型、VMD 增强、WOA 优化与完整组合。",
            tideTitle: "TiDE 模型组",
            tideDesc: "评估多尺度分解与智能优化在 TiDE 预测中的增益。",
            config: "实验配置",
            optimizationEyebrow: "HYPERPARAMETER SEARCH",
            optimizationTitle: "WOA 超参数优化过程",
            optimizationDesc: "以三维散点视图观察不同超参数组合与目标函数之间的关系。",
            woaTsmixer: "TSMixer 参数搜索",
            woaTsmixerDesc: "WOA 在 TSMixer 参数空间中的搜索分布。",
            woaTide: "TiDE 参数搜索",
            woaTideDesc: "WOA 在 TiDE 参数空间中的搜索分布。",
            userTitle: "预测系统",
            userDesc: "当前账号已成功登录。管理员可进入完整的数据分析与模型评估工作台。"
        },
        en: {
            brandCaption: "Boiler Fouling Intelligence",
            navHome: "Overview",
            navCleaning: "Data Cleaning",
            navPrediction: "Ablation Study",
            navOptimization: "Optimization",
            topTitle: "Industrial Time-Series Intelligence",
            logout: "Sign out",
            language: "中文",
            loginEyebrow: "PSBAF · INDUSTRIAL AI",
            loginHeroTitle: "Make boiler fouling trends predictable",
            loginHeroDesc: "Signal decomposition, deep time-series forecasting and intelligent optimization combined into one clear and dependable workflow.",
            loginTitle: "Welcome back",
            adminLoginTitle: "Administrator sign in",
            loginSubtitle: "Sign in to access the boiler fouling prediction and analytics platform.",
            username: "Username",
            password: "Password",
            usernamePlaceholder: "Enter username",
            passwordPlaceholder: "Enter password",
            loginButton: "Sign in",
            adminEntry: "Administrator portal",
            userEntry: "User portal",
            demoHint: "Local demo account",
            heroKicker: "POWER PLANT INTELLIGENCE",
            heroTitle: "Boiler Ash Fouling Prediction System",
            heroDesc: "An end-to-end workflow integrating wavelet denoising, VMD decomposition, TSMixer / TiDE forecasting and WOA-based optimization.",
            featureCleaning: "Signal preprocessing",
            featureCleaningDesc: "Improve cleaning-factor signal quality with soft and hard wavelet thresholding.",
            featurePrediction: "Multi-model forecasting",
            featurePredictionDesc: "Compare TSMixer, TiDE and their VMD-enhanced forecasting variants.",
            featureOptimization: "Intelligent optimization",
            featureOptimizationDesc: "Search key hyperparameters with WOA and visualize the optimization landscape.",
            cleaningEyebrow: "SIGNAL PREPROCESSING",
            cleaningTitle: "Data Cleaning & Quality Assessment",
            cleaningDesc: "Compare soft and hard wavelet thresholding with residual-energy and smoothness metrics.",
            softTitle: "Soft-threshold Denoising",
            softDesc: "Preserves continuity and produces a smoother cleaning-factor trend.",
            hardTitle: "Hard-threshold Denoising",
            hardDesc: "Removes coefficients below the threshold to emphasize the main signal structure.",
            rer: "Residual energy ratio",
            smooth: "Smoothness index",
            predictionEyebrow: "ABLATION STUDY",
            predictionTitle: "Forecasting Model Ablation Study",
            predictionDesc: "Measure how VMD decomposition and WOA optimization contribute to TSMixer and TiDE.",
            tsmixerTitle: "TSMixer Model Family",
            tsmixerDesc: "Compare the baseline, VMD-enhanced, WOA-optimized and complete configurations.",
            tideTitle: "TiDE Model Family",
            tideDesc: "Evaluate the gains from multi-scale decomposition and intelligent optimization.",
            config: "Configuration",
            optimizationEyebrow: "HYPERPARAMETER SEARCH",
            optimizationTitle: "WOA Hyperparameter Optimization",
            optimizationDesc: "Explore relationships between hyperparameter combinations and objective values in 3D.",
            woaTsmixer: "TSMixer Search Space",
            woaTsmixerDesc: "WOA search distribution across the TSMixer parameter space.",
            woaTide: "TiDE Search Space",
            woaTideDesc: "WOA search distribution across the TiDE parameter space.",
            userTitle: "Prediction System",
            userDesc: "You are signed in. Administrators can access the complete analytics and model evaluation workspace."
        }
    };

    function applyLanguage(lang) {
        const dict = dictionaries[lang] || dictionaries.zh;
        document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
        document.querySelectorAll("[data-i18n]").forEach(function (el) {
            const value = dict[el.dataset.i18n];
            if (value !== undefined) {
                if (el.tagName === "INPUT") el.value = value;
                else el.textContent = value;
            }
        });
        document.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
            const value = dict[el.dataset.i18nPlaceholder];
            if (value !== undefined) el.placeholder = value;
        });
        localStorage.setItem("psbaf-language", lang);
    }

    document.addEventListener("DOMContentLoaded", function () {
        let lang = localStorage.getItem("psbaf-language") || "zh";
        applyLanguage(lang);
        document.querySelectorAll(".lang-switch").forEach(function (button) {
            button.addEventListener("click", function () {
                lang = lang === "zh" ? "en" : "zh";
                applyLanguage(lang);
            });
        });
    });
})();
