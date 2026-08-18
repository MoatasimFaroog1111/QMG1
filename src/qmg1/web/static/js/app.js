import { QmgApiClient } from "./api.js";
import { DashboardController } from "./dashboard.js";

const api = new QmgApiClient();
const dashboard = new DashboardController(api);

void dashboard.init();
