// 病历记录接口
export interface MedicalRecord {
  id: number;
  symptoms: string;
  diagnosis: string;
  prescription: string;
  created_at: string;
}

export interface User {
  userId: number;
  username: string;
  name: string;
  age: number;
  gender: string;
  phone: string;
  created_at?: string;
  records?: MedicalRecord[];
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  password: string;
  name: string;
  age: number;
  gender: string;
  phone: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  data: {
    token: string;
    user: User;
  };
}

export interface RegisterResponse {
  success: boolean;
  message: string;
}

export interface ApiError {
  success: false;
  message: string;
  errors?: Array<{
    field: string;
    message: string;
  }>;
}